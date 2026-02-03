import os
from typing import Any, Dict, Optional

import requests

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3-flash-preview"


class GeminiError(RuntimeError):
    pass


def generate_content(
    user_text: str,
    *,
    system_text: Optional[str] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError("GEMINI_API_KEY is not set")

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
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
