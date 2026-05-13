# Codex Cut

Codex Cut is a Codex-native video production plugin. The product wedge is simple:

> Edit videos by editing the code, timeline, captions, assets, and exports from inside Codex.

This is not a browser video editor with an AI side panel. It is a file-first production environment where the output is normal project material: timeline JSON, Remotion source, captions, audio assets, thumbnails, platform exports, and QA reports.

## MVP Commands

- `/video import ./raw-demo.mp4`
- `/video find-best-moments`
- `/video cut 45s trailer for Product Hunt`
- `/video add captions`
- `/video add punchy zooms`
- `/video add beat-synced transitions`
- `/video make youtube thumbnail`
- `/video export twitter`
- `/video export youtube`
- `/video export producthunt`
- `/video qa`

## Architecture

- `runtime/timeline-schema.json`: canonical Timeline IR.
- `runtime/platform-presets.json`: export sizes, codecs, bitrate, safe areas, caption defaults.
- `runtime/render-remotion.ts`: Remotion composition helpers from Timeline IR.
- `runtime/ffmpeg-tools.ts`: FFmpeg command builders for probing, trimming, transcoding, captions, and frames.
- `runtime/qa-frames.ts`: visual QA heuristics for blank frames, crop, contrast, and caption safe areas.
- `runtime/asset-memory-schema.json`: reusable brand, caption, audio, thumbnail, and export preferences.
- `templates/`: starter timelines for launch trailers, app demos, shorts, and Product Hunt clips.
- `commands/`: command playbooks for Codex behavior.

## First Production Target

The first high-value workflow is:

```txt
/video make launch trailer from this app
```

Expected behavior:

1. Inspect the local app and repo.
2. Capture or import product footage.
3. Draft a 35-45 second script.
4. Build a Timeline IR.
5. Generate editable Remotion code.
6. Add captions, zooms, transitions, and audio markers.
7. Export Product Hunt, X, YouTube Shorts, LinkedIn, and YouTube variants.
8. Run visual QA for blank frames, crop, overlap, contrast, and unreadable captions.
9. Save final `.mp4`, `.srt`, thumbnail, post copy, and QA report.
