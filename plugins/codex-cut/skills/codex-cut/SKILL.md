# Codex Cut

Use this skill when the user asks for Codex-native video editing, launch trailers, timeline editing, Remotion rendering, FFmpeg exports, captioning, video QA, or platform-specific social video packages.

## Core Principle

Codex is the editor. Keep every production decision in normal files:

- `timeline.json` for clips, captions, overlays, zooms, transitions, audio, and exports.
- Remotion `.tsx` for editable generated video.
- `.srt` or `.vtt` for captions.
- `.mp4`, `.png`, `.wav`, and `.json` outputs in a predictable export folder.

Do not bury creative state in hidden UI state.

## Workflow

1. Locate or create a project folder such as `video/<slug>/`.
2. Create or update `timeline.json` using `runtime/timeline-schema.json`.
3. Use `runtime/platform-presets.json` for aspect ratio, safe areas, codecs, and export settings.
4. Generate editable Remotion code with the structure from `runtime/render-remotion.ts`.
5. Use FFmpeg command builders from `runtime/ffmpeg-tools.ts` for import, trim, transcode, audio normalize, frame extraction, and caption burn-in.
6. Run visual QA with the checks in `runtime/qa-frames.ts`.
7. Produce final exports plus a concise `qa-report.md`.

## Required Guardrails

- Preserve source files. Write derived media to an `exports/` or `build/` folder.
- Confirm before destructive trims or overwrites.
- Keep platform captions inside safe areas.
- Extract QA frames before calling an export done.
- Flag blank frames, low-contrast captions, cropped faces/UI, clipped product text, and audio peaks.
- Prefer short, bounded outputs first: 35-45 second launch trailers, 15-30 second shorts, or one chapter at a time.

## `/video make launch trailer from this app`

For this command:

1. Inspect the app and repo surface.
2. Identify the product promise, audience, key workflow, and proof points.
3. Build a short script with hook, demo, proof, CTA.
4. Create a Timeline IR with clips, captions, zooms, overlays, and export targets.
5. Generate editable Remotion source.
6. Export at least `producthunt`, `x_landscape`, and `youtube_short` variants when render dependencies are present.
7. QA screenshots at start, mid-scene, transition points, and end card.

## Output Shape

When done, report:

- timeline path
- generated Remotion path
- exported files
- QA findings
- remaining blockers

