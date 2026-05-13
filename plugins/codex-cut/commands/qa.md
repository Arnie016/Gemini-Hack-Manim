# /video qa

Check whether a video is visually and structurally ready to share.

## Usage

```txt
/video qa
/video qa ./exports/youtube-short/demo.mp4
```

## Checks

- blank or near-blank frames
- unreadable or low-contrast captions
- captions outside safe areas
- cropped product UI, faces, or title text
- abrupt audio peaks
- missing thumbnail
- missing post copy
- export dimensions mismatch

## Output

- `qa/frames/*.png`
- `qa/qa-report.md`
- `qa/findings.json`

