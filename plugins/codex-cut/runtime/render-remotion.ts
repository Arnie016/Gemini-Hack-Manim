export type Timeline = {
  version: "0.1";
  project: {
    slug: string;
    title: string;
    description?: string;
    brand?: {
      colors?: string[];
      fontFamily?: string;
      logo?: string;
    };
  };
  duration: number;
  fps?: number;
  sources?: Array<{ id: string; path: string; type: string; duration?: number }>;
  tracks: Array<{
    id: string;
    type: "video" | "audio" | "caption" | "overlay" | "effect";
    items: TimelineItem[];
  }>;
  exports: Array<{ preset: string; filename?: string; burnCaptions?: boolean; thumbnailAt?: number }>;
};

export type TimelineItem = {
  id: string;
  kind: "clip" | "caption" | "title" | "image" | "zoom" | "transition" | "sfx" | "music" | "shape";
  start: number;
  duration: number;
  sourceId?: string;
  text?: string;
  style?: Record<string, unknown>;
  transform?: {
    x?: number;
    y?: number;
    scale?: number;
    rotate?: number;
    opacity?: number;
  };
};

export function compositionSettings(timeline: Timeline, width: number, height: number) {
  const fps = timeline.fps ?? 30;
  return {
    id: timeline.project.slug,
    width,
    height,
    fps,
    durationInFrames: Math.ceil(timeline.duration * fps)
  };
}

export function secondsToFrames(seconds: number, fps: number): number {
  return Math.max(0, Math.round(seconds * fps));
}

export function sceneItemsAt(timeline: Timeline, timeSeconds: number): TimelineItem[] {
  return timeline.tracks
    .flatMap((track) => track.items)
    .filter((item) => timeSeconds >= item.start && timeSeconds < item.start + item.duration)
    .sort((a, b) => a.start - b.start);
}

export function buildRemotionComponentSource(timeline: Timeline): string {
  const serialized = JSON.stringify(timeline, null, 2);
  return `import React from "react";
import {AbsoluteFill, Sequence, spring, useCurrentFrame, useVideoConfig} from "remotion";

const timeline = ${serialized} as const;

export function Video() {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;

  return (
    <AbsoluteFill style={{background: "#070B12", color: "white", fontFamily: "${timeline.project.brand?.fontFamily ?? "Inter, system-ui, sans-serif"}"}}>
      {timeline.tracks.flatMap((track) =>
        track.items.map((item) => (
          <Sequence key={item.id} from={Math.round(item.start * fps)} durationInFrames={Math.round(item.duration * fps)}>
            <TimelineItem item={item} fps={fps} time={time} />
          </Sequence>
        ))
      )}
    </AbsoluteFill>
  );
}

function TimelineItem({item, fps, time}: {item: any; fps: number; time: number}) {
  const frame = useCurrentFrame();
  const entrance = spring({frame, fps, config: {damping: 18, stiffness: 130}});
  const opacity = item.transform?.opacity ?? 1;
  const scale = item.transform?.scale ?? 1;
  const x = item.transform?.x ?? 0;
  const y = item.transform?.y ?? 0;

  if (item.kind === "caption" || item.kind === "title") {
    return (
      <div style={{
        position: "absolute",
        left: "8%",
        right: "8%",
        bottom: item.kind === "title" ? "42%" : "9%",
        transform: \`translate(\${x}px, \${y}px) scale(\${scale * (0.96 + entrance * 0.04)})\`,
        opacity,
        fontSize: item.kind === "title" ? 72 : 44,
        fontWeight: 800,
        lineHeight: 1.06,
        textShadow: "0 3px 18px rgba(0,0,0,.7)"
      }}>
        {item.text}
      </div>
    );
  }

  if (item.kind === "shape") {
    return (
      <div style={{
        position: "absolute",
        inset: "12%",
        border: "2px solid rgba(107,168,255,.75)",
        borderRadius: 18,
        transform: \`translate(\${x}px, \${y}px) scale(\${scale})\`,
        opacity
      }} />
    );
  }

  return null;
}
`;
}

