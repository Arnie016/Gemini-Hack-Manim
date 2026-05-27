# NorthStar Science Backtesting

NorthStar now has a deterministic science concept suite for checking whether the app can keep producing real Manim scenes at scale. This is separate from one-off smoke tests: it generates many explicit physics, math, biology, chemistry, climate, AI, and systems prompts, writes editable Manim code for each one, and can optionally render and score MP4 output.

## Fast Code-Only Suite

Use this first when changing prompt flow, templates, file handling, or CLI behavior:

```bash
python3 -m backend.cli backtest-science --limit 100 --out-dir work/backtests/science-100-code
```

This writes:

- `work/backtests/science-100-code/report.md`
- `work/backtests/science-100-code/report.json`
- one `scene.py` file per concept

It does not call OpenAI or Manim. It is meant to confirm the app has a broad, specific science concept corpus and can generate repeatable baseline Manim code.

## Real MP4 Render Suite

Use a small render suite before shipping renderer changes:

```bash
python3 -m backend.cli backtest-science \
  --limit 5 \
  --render \
  --seconds 12 \
  --quality pql \
  --out-dir work/backtests/science-render-5
```

Use a specific Manim runtime when needed:

```bash
python3 -m backend.cli backtest-science \
  --limit 10 \
  --render \
  --seconds 18 \
  --quality pql \
  --manim-py /Users/arnav/Desktop/Gemini-Hack-Manim-main/.venv/bin/python \
  --out-dir work/backtests/science-render-10
```

For a larger overnight local run:

```bash
python3 -m backend.cli backtest-science \
  --limit 100 \
  --render \
  --seconds 20 \
  --quality pql \
  --out-dir work/backtests/science-render-100
```

## Scoring

Each rendered MP4 is scored from 0 to 100 using local checks:

- MP4 file exists.
- Duration is not far below or above target.
- File size is plausible.
- `ffprobe` detects a video stream.
- Manim logs do not contain obvious errors.
- Manim emitted the expected `Rendered GeneratedScene` success marker.

The report shows pass/fail, score, duration, file size, paths, and issues for every concept. A score below 70 means the render is not production-trustworthy for that concept.

## What This Catches

- Manim runtime regressions.
- FFmpeg or `ffprobe` missing.
- Code-generation syntax failures.
- Extremely short or empty videos.
- Silent missing-output failures.
- Broad concept coverage gaps across science domains.

## What This Does Not Yet Prove

This suite validates deterministic Manim generation and rendering. It does not yet validate the full OpenAI planning/chat flow across 100 concepts. The next layer should run API-level workflows:

```text
prompt -> /api/plan -> /api/approve -> async render -> MP4 score -> gallery/share artifact check
```

That API harness should reuse the same concept corpus from `backend/science_concepts.py`, then compare AI-generated outputs against this deterministic baseline.
