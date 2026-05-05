import base64
import os
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"
DEFAULT_IMAGE_ASPECT = "9:16"
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-5"
DEFAULT_OPENAI_FALLBACK_MODEL = "gpt-5-mini"


def _openai_timeout_seconds() -> float:
    raw = (os.getenv("OPENAI_TIMEOUT_SECONDS") or "180").strip()
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 180.0


def _openai_fallback_model(primary_model: str) -> Optional[str]:
    fallback = (os.getenv("OPENAI_FALLBACK_MODEL") or DEFAULT_OPENAI_FALLBACK_MODEL).strip()
    if not fallback or fallback == primary_model:
        return None
    return fallback


class GeminiError(RuntimeError):
    pass


def _provider_for_model(model: Optional[str], provider: Optional[str]) -> str:
    raw_provider = (provider or "").strip().lower()
    if raw_provider in {"openai", "gemini"}:
        return raw_provider
    raw_model = (model or "").strip().lower()
    if raw_model.startswith("gemini-"):
        return "gemini"
    return "openai"


def _google_schema_to_json_schema(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_google_schema_to_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.lower()
            continue
        if key in {"properties", "$defs", "definitions"} and isinstance(value, dict):
            out[key] = {
                str(k): _google_schema_to_json_schema(v) for k, v in value.items()
            }
            continue
        if key == "items":
            out[key] = _google_schema_to_json_schema(value)
            continue
        out[key] = _google_schema_to_json_schema(value)
    return out


def _openai_response_format(
    generation_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not generation_config:
        return None
    if generation_config.get("response_mime_type") != "application/json":
        return None
    schema = generation_config.get("response_schema")
    if not schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "northstar_response",
            "strict": False,
            "schema": _google_schema_to_json_schema(schema),
        },
    }


def generate_content(
    user_text: str,
    *,
    system_text: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    selected_provider = _provider_for_model(model, provider)
    if selected_provider == "openai":
        return _generate_openai_content(
            user_text,
            system_text=system_text,
            generation_config=generation_config,
            api_key=api_key,
            model=model,
        )

    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set")

    model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    url = f"{BASE_URL}/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_text}],
            }
        ]
    }

    if system_text:
        payload["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": system_text}],
        }

    if generation_config:
        payload["generationConfig"] = generation_config

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise GeminiError(f"Gemini request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise GeminiError(f"Gemini error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected Gemini response: {data}") from exc


def _generate_openai_content(
    user_text: str,
    *,
    system_text: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise GeminiError("OPENAI_API_KEY is not set")

    model = model or os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "store": False,
    }
    response_format = _openai_response_format(generation_config)
    if response_format:
        payload["response_format"] = response_format

    timeout = _openai_timeout_seconds()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.Timeout as exc:
        fallback = _openai_fallback_model(model)
        if not fallback:
            raise GeminiError(f"OpenAI request failed: {exc}") from exc
        payload["model"] = fallback
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as fallback_exc:
            raise GeminiError(
                f"OpenAI request timed out for {model}; fallback {fallback} also failed: {fallback_exc}"
            ) from fallback_exc
    except requests.RequestException as exc:
        raise GeminiError(f"OpenAI request failed: {exc}") from exc

    if resp.status_code >= 400:
        fallback = _openai_fallback_model(model)
        if fallback and resp.status_code in {408, 409, 429, 500, 502, 503, 504}:
            payload["model"] = fallback
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            except requests.RequestException as fallback_exc:
                raise GeminiError(
                    f"OpenAI error {resp.status_code} for {model}; fallback {fallback} also failed: {fallback_exc}"
                ) from fallback_exc
            if resp.status_code >= 400:
                raise GeminiError(f"OpenAI error {resp.status_code}: {resp.text}")
            model = fallback
        else:
            raise GeminiError(f"OpenAI error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected OpenAI response: {data}") from exc
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        ).strip()
    raise GeminiError(f"Unexpected OpenAI content: {data}")


def generate_image(
    prompt: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bytes:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set")

    model = model or os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)
    url = f"{BASE_URL}/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    payload: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    aspect = os.getenv("GEMINI_IMAGE_ASPECT", DEFAULT_IMAGE_ASPECT)
    payload["generationConfig"] = {"imageConfig": {"aspectRatio": aspect}}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as exc:
        raise GeminiError(f"Gemini image request failed: {exc}") from exc

    if resp.status_code >= 400:
        raise GeminiError(f"Gemini image error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError(f"Unexpected Gemini image response: {data}") from exc

    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"])

    raise GeminiError(f"No inline image data in response: {data}")
