from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .backtest import run_local_backtest


DEFAULT_BASE_URL = "http://127.0.0.1:8000"
COOKIE_PATH = Path(os.getenv("NORTHSTAR_CLI_COOKIE", "~/.northstar/cookies.json")).expanduser()


class CliError(RuntimeError):
    pass


def _base_url(args: argparse.Namespace) -> str:
    return str(getattr(args, "base_url", "") or os.getenv("NORTHSTAR_URL") or DEFAULT_BASE_URL).rstrip("/")


def _session() -> requests.Session:
    session = requests.Session()
    if COOKIE_PATH.exists():
        try:
            cookies = json.loads(COOKIE_PATH.read_text(encoding="utf-8"))
            if isinstance(cookies, dict):
                session.cookies.update(cookies)
        except Exception:
            pass
    return session


def _save_cookies(session: requests.Session) -> None:
    COOKIE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_PATH.write_text(json.dumps(requests.utils.dict_from_cookiejar(session.cookies), indent=2), encoding="utf-8")


def _request_json(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 240,
) -> Dict[str, Any]:
    session = _session()
    url = f"{_base_url(args)}{path}"
    try:
        resp = session.request(method, url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise CliError(
            f"Could not reach NorthStar at {url}.\n"
            f"Start it with: python -m backend.cli serve\n"
            f"Details: {exc}"
        ) from exc
    _save_cookies(session)
    try:
        data = resp.json()
    except ValueError as exc:
        raise CliError(f"NorthStar returned non-JSON response ({resp.status_code}): {resp.text[:500]}") from exc
    if resp.status_code >= 400 or data.get("ok") is False:
        raise CliError(str(data.get("error") or data.get("message") or f"HTTP {resp.status_code}"))
    return data


def _print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _plan_payload(args: argparse.Namespace, prompt: str) -> Dict[str, Any]:
    return {
        "idea": prompt,
        "audience": args.audience,
        "tone": args.tone,
        "style": args.style,
        "pace": args.pace,
        "color_palette": args.palette,
        "include_equations": not args.no_equations,
        "include_graphs": not args.no_graphs,
        "include_narration": not args.no_narration,
        "target_seconds": args.seconds,
        "max_scenes": args.max_scenes,
        "aspect_ratio": args.aspect,
        "model": args.model,
    }


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    data = _request_json(args, "GET", "/api/healthz", timeout=30)
    _print_json(data)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        raise CliError("Prompt is required.")
    data = _request_json(args, "POST", "/api/plan", payload=_plan_payload(args, prompt), timeout=240)
    if args.out:
        Path(args.out).write_text(data.get("plan_text") or json.dumps(data.get("plan", {}), indent=2), encoding="utf-8")
    _print_json(data)
    return 0


def _wait_for_job(args: argparse.Namespace, job_id: str) -> Dict[str, Any]:
    started = time.time()
    last_line = ""
    while True:
        data = _request_json(args, "GET", f"/api/jobs/{job_id}", timeout=30)
        line = f"{data.get('status')}:{data.get('step')} {data.get('message') or data.get('error') or ''}".strip()
        if line != last_line:
            print(line, file=sys.stderr)
            last_line = line
        if data.get("status") in {"done", "failed"}:
            return data
        if time.time() - started > args.timeout:
            raise CliError(f"Timed out waiting for job {job_id}.")
        time.sleep(args.poll)


def cmd_status(args: argparse.Namespace) -> int:
    data = _request_json(args, "GET", f"/api/jobs/{args.job_id}", timeout=30)
    _print_json(data)
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    plan_text = Path(args.plan_file).read_text(encoding="utf-8") if args.plan_file else None
    if not plan_text:
        status = _request_json(args, "GET", f"/api/jobs/{args.job_id}", timeout=30)
        plan_text = json.dumps(status.get("plan") or {}, indent=2)
    payload = {
        "job_id": args.job_id,
        "plan_text": plan_text,
        "quality": args.quality,
        "aspect_ratio": args.aspect,
        "model": args.model,
    }
    data = _request_json(args, "POST", "/api/approve", payload=payload, timeout=120)
    if args.wait:
        data = _wait_for_job(args, args.job_id)
    _print_json(data)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        raise CliError("Prompt is required.")
    plan = _request_json(args, "POST", "/api/plan", payload=_plan_payload(args, prompt), timeout=240)
    job_id = plan["job_id"]
    payload = {
        "job_id": job_id,
        "plan_text": plan.get("plan_text") or json.dumps(plan.get("plan", {}), indent=2),
        "quality": args.quality,
        "aspect_ratio": args.aspect,
        "model": args.model,
    }
    _request_json(args, "POST", "/api/approve", payload=payload, timeout=120)
    data = _wait_for_job(args, job_id) if args.wait else {"ok": True, "job_id": job_id, "status": "queued"}
    _print_json(data)
    return 0


def cmd_voiceover(args: argparse.Namespace) -> int:
    script = Path(args.script_file).read_text(encoding="utf-8") if args.script_file else args.script
    payload = {
        "provider": args.provider,
        "voice_id": args.voice,
        "model_id": args.model,
        "script_text": script,
        "include_chat_context": False,
        "use_gemini_script": args.draft_script,
    }
    data = _request_json(args, "POST", f"/api/jobs/{args.job_id}/voiceover", payload=payload, timeout=240)
    _print_json(data)
    return 0


def cmd_backtest_science(args: argparse.Namespace) -> int:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("work") / "backtests" / f"science-{stamp}"
    report = run_local_backtest(
        out_dir=out_dir,
        limit=args.limit,
        family=args.family,
        render=args.render,
        manim_py=args.manim_py,
        quality=args.quality,
        seconds=args.seconds,
        aspect_ratio=args.aspect,
    )
    if args.json:
        _print_json(report)
        return 0

    print(f"NorthStar science backtest: {report['mode']}")
    score_label = report["average_score"] if report["average_score"] is not None else "n/a"
    print(f"Concepts: {report['passed']}/{report['total']} passed, average score {score_label}")
    print(f"Report: {Path(report['out_dir']) / 'report.md'}")
    print(f"JSON: {Path(report['out_dir']) / 'report.json'}")
    if report["failed"]:
        for item in report["results"]:
            if not item.get("ok"):
                print(f"- FAIL {item['topic']}: {', '.join(item.get('issues') or [])}")
    return 0 if report["ok"] or not args.fail_on_render_score else 1


def _add_common_generation_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Text model override, e.g. gpt-5-mini.")
    parser.add_argument("--seconds", type=float, default=None, help="Target duration in seconds.")
    parser.add_argument("--max-scenes", type=int, default=None, help="Maximum scene count.")
    parser.add_argument("--aspect", default="9:16", help="Aspect ratio, e.g. 9:16 or 16:9.")
    parser.add_argument("--audience", default="general")
    parser.add_argument("--tone", default="epic")
    parser.add_argument("--style", default="cinematic")
    parser.add_argument("--pace", default="medium")
    parser.add_argument("--palette", default="cool")
    parser.add_argument("--no-equations", action="store_true")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--no-narration", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="northstar", description="CLI for NorthStar Manim animation workflows.")
    parser.add_argument("--base-url", default=None, help=f"NorthStar backend URL. Default: {DEFAULT_BASE_URL}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the local NorthStar backend.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    health = sub.add_parser("health", help="Check backend health.")
    health.set_defaults(func=cmd_health)

    plan = sub.add_parser("plan", help="Create an editable scene plan.")
    _add_common_generation_flags(plan)
    plan.add_argument("--out", help="Write plan JSON to a file.")
    plan.add_argument("prompt", nargs=argparse.REMAINDER)
    plan.set_defaults(func=cmd_plan)

    approve = sub.add_parser("approve", help="Approve an existing job plan and render it.")
    approve.add_argument("job_id")
    approve.add_argument("--plan-file", help="Plan JSON file. Defaults to the job's saved plan.")
    approve.add_argument("--quality", default="pql", choices=["pql", "pqm", "pqh", "low", "medium", "high"])
    approve.add_argument("--aspect", default="9:16")
    approve.add_argument("--model", default=None)
    approve.add_argument("--wait", action="store_true")
    approve.add_argument("--poll", type=float, default=2.0)
    approve.add_argument("--timeout", type=float, default=900)
    approve.set_defaults(func=cmd_approve)

    render = sub.add_parser("render", help="Plan, approve, render, and poll a prompt.")
    _add_common_generation_flags(render)
    render.add_argument("--quality", default="pql", choices=["pql", "pqm", "pqh", "low", "medium", "high"])
    render.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    render.add_argument("--poll", type=float, default=2.0)
    render.add_argument("--timeout", type=float, default=900)
    render.add_argument("prompt", nargs=argparse.REMAINDER)
    render.set_defaults(func=cmd_render)

    status = sub.add_parser("status", help="Show job status.")
    status.add_argument("job_id")
    status.set_defaults(func=cmd_status)

    voice = sub.add_parser("voiceover", help="Add OpenAI or ElevenLabs narration to a rendered job.")
    voice.add_argument("job_id")
    voice.add_argument("--provider", default="openai", choices=["openai", "elevenlabs"])
    voice.add_argument("--voice", default="marin", help="OpenAI voice like marin/cedar or ElevenLabs voice ID.")
    voice.add_argument("--model", default="gpt-4o-mini-tts", help="OpenAI TTS model or ElevenLabs model.")
    voice.add_argument("--script", default=None)
    voice.add_argument("--script-file", default=None)
    voice.add_argument("--draft-script", action="store_true", help="Let the text model draft narration from the saved plan.")
    voice.set_defaults(func=cmd_voiceover)

    backtest = sub.add_parser(
        "backtest-science",
        help="Generate, optionally render, and score a suite of science Manim concepts.",
    )
    backtest.add_argument("--limit", type=int, default=10, help="Number of concepts to test. Use 100 for the broad suite.")
    backtest.add_argument("--family", default=None, help="Optional family filter, e.g. Optics, Mechanics, Quantum.")
    backtest.add_argument("--render", action="store_true", help="Actually render MP4s with Manim. Omit for fast code-only generation.")
    backtest.add_argument("--quality", default="pql", choices=["pql", "pqm", "pqh", "low", "medium", "high"])
    backtest.add_argument("--seconds", type=float, default=12.0, help="Target seconds per deterministic test video.")
    backtest.add_argument("--aspect", default="9:16", help="Aspect ratio for generated scenes, e.g. 9:16 or 16:9.")
    backtest.add_argument("--manim-py", default=None, help="Python executable with Manim installed.")
    backtest.add_argument("--out-dir", default=None, help="Output directory for generated code, MP4s, logs, and reports.")
    backtest.add_argument("--json", action="store_true", help="Print the full JSON report.")
    backtest.add_argument(
        "--fail-on-render-score",
        action="store_true",
        help="Exit non-zero when any rendered concept scores below the pass threshold.",
    )
    backtest.set_defaults(func=cmd_backtest_science)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except CliError as exc:
        print(f"northstar: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
