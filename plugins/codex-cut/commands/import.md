# /video import

Import source footage into a file-first video project.

## Usage

```txt
/video import ./raw-demo.mp4
/video import ./recordings --project launch-demo
```

## Behavior

1. Create `video/<project>/sources/` if needed.
2. Copy or reference source media without modifying originals.
3. Probe media with FFmpeg/ffprobe.
4. Add source metadata to `timeline.json`.
5. Extract low-cost contact sheet frames for review.

## Output

- `video/<project>/timeline.json`
- `video/<project>/sources/manifest.json`
- `video/<project>/qa/contact-sheet.json`

