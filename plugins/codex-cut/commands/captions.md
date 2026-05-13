# /video captions

Create, style, and validate captions.

## Usage

```txt
/video add captions
/video captions burn in for youtube-short
```

## Behavior

1. Generate or import transcript segments.
2. Split captions into readable lines.
3. Apply platform-safe placement from `platform-presets.json`.
4. Export `.srt` and optional burn-in captions.
5. QA contrast, line length, timing, and overlap.

## Output

- `captions/main.srt`
- `captions/main.vtt`
- updated timeline caption tracks

