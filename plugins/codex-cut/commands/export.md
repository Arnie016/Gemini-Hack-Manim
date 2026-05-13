# /video export

Render platform-specific exports from a Timeline IR.

## Usage

```txt
/video export producthunt
/video export youtube-short x linkedin
```

## Behavior

1. Load `timeline.json`.
2. Validate requested presets.
3. Render through Remotion when composition source exists.
4. Use FFmpeg for transcode, normalization, thumbnails, and caption burn-in.
5. Save export metadata and checksums.

## Output

- `exports/<platform>/<project>.mp4`
- `exports/<platform>/thumbnail.png`
- `exports/<platform>/post-copy.md`
- `exports/<platform>/metadata.json`

