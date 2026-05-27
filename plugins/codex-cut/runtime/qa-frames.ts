export type Rect = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type QaFinding = {
  level: "info" | "warning" | "error";
  code: string;
  message: string;
  timestamp?: number;
};

export type FrameStats = {
  timestamp: number;
  averageLuma: number;
  lumaStdDev: number;
  textRects?: Rect[];
  safeArea?: Rect;
};

export function detectBlankFrames(frames: FrameStats[]): QaFinding[] {
  return frames
    .filter((frame) => frame.averageLuma < 6 || frame.lumaStdDev < 2)
    .map((frame) => ({
      level: "error",
      code: "blank_frame",
      timestamp: frame.timestamp,
      message: "Frame appears blank or visually flat. Check source clip, transition, or render timing."
    }));
}

export function detectCaptionSafeAreaIssues(frames: FrameStats[]): QaFinding[] {
  const findings: QaFinding[] = [];

  for (const frame of frames) {
    if (!frame.textRects || !frame.safeArea) continue;
    for (const rect of frame.textRects) {
      if (!contains(frame.safeArea, rect)) {
        findings.push({
          level: "warning",
          code: "caption_safe_area",
          timestamp: frame.timestamp,
          message: "Text or caption falls outside the platform safe area."
        });
      }
    }
  }

  return findings;
}

export function detectLowContrastCaption(frame: FrameStats, contrastRatio: number): QaFinding | null {
  if (contrastRatio >= 4.5) return null;
  return {
    level: "warning",
    code: "caption_contrast",
    timestamp: frame.timestamp,
    message: `Caption contrast ratio ${contrastRatio.toFixed(2)} is below the 4.5 readability target.`
  };
}

export function detectDurationMismatch(duration: number, maxDuration: number, preset: string): QaFinding | null {
  if (duration <= maxDuration) return null;
  return {
    level: "warning",
    code: "duration_limit",
    message: `${preset} export is ${duration}s but preset max is ${maxDuration}s.`
  };
}

export function contains(outer: Rect, inner: Rect): boolean {
  return (
    inner.x >= outer.x &&
    inner.y >= outer.y &&
    inner.x + inner.width <= outer.x + outer.width &&
    inner.y + inner.height <= outer.y + outer.height
  );
}

export function defaultSampleTimestamps(duration: number): number[] {
  const anchors = [0.5, 0.15, 0.33, 0.5, 0.67, 0.85, Math.max(0.5, duration - 0.5)];
  return Array.from(
    new Set(
      anchors
        .map((value) => (value < 1 ? value * duration : value))
        .map((value) => Number(Math.min(Math.max(value, 0), duration).toFixed(2)))
    )
  );
}

