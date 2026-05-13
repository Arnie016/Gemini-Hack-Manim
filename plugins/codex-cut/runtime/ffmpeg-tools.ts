export type PlatformPreset = {
  width: number;
  height: number;
  fps: number;
  videoCodec: string;
  audioCodec: string;
  videoBitrate: string;
};

export type CommandSpec = {
  command: "ffmpeg" | "ffprobe";
  args: string[];
};

export function probe(inputPath: string): CommandSpec {
  return {
    command: "ffprobe",
    args: [
      "-v",
      "error",
      "-show_format",
      "-show_streams",
      "-of",
      "json",
      inputPath
    ]
  };
}

export function extractFrames(inputPath: string, outputPattern: string, timestamps: number[]): CommandSpec[] {
  return timestamps.map((timestamp) => ({
    command: "ffmpeg",
    args: [
      "-y",
      "-ss",
      String(timestamp),
      "-i",
      inputPath,
      "-frames:v",
      "1",
      outputPattern.replace("%t", String(timestamp).replace(".", "_"))
    ]
  }));
}

export function trimClip(inputPath: string, outputPath: string, start: number, duration: number): CommandSpec {
  return {
    command: "ffmpeg",
    args: [
      "-y",
      "-ss",
      String(start),
      "-i",
      inputPath,
      "-t",
      String(duration),
      "-c",
      "copy",
      outputPath
    ]
  };
}

export function normalizeAudio(inputPath: string, outputPath: string): CommandSpec {
  return {
    command: "ffmpeg",
    args: [
      "-y",
      "-i",
      inputPath,
      "-af",
      "loudnorm=I=-16:TP=-1.5:LRA=11",
      "-c:v",
      "copy",
      outputPath
    ]
  };
}

export function transcode(inputPath: string, outputPath: string, preset: PlatformPreset): CommandSpec {
  return {
    command: "ffmpeg",
    args: [
      "-y",
      "-i",
      inputPath,
      "-vf",
      `scale=${preset.width}:${preset.height}:force_original_aspect_ratio=decrease,pad=${preset.width}:${preset.height}:(ow-iw)/2:(oh-ih)/2,fps=${preset.fps}`,
      "-c:v",
      preset.videoCodec,
      "-b:v",
      preset.videoBitrate,
      "-pix_fmt",
      "yuv420p",
      "-c:a",
      preset.audioCodec,
      "-movflags",
      "+faststart",
      outputPath
    ]
  };
}

export function burnCaptions(inputPath: string, srtPath: string, outputPath: string): CommandSpec {
  return {
    command: "ffmpeg",
    args: [
      "-y",
      "-i",
      inputPath,
      "-vf",
      `subtitles=${escapeFilterPath(srtPath)}:force_style='Fontsize=42,Outline=2,Shadow=1,Alignment=2,MarginV=96'`,
      "-c:a",
      "copy",
      outputPath
    ]
  };
}

function escapeFilterPath(path: string): string {
  return path.replace(/\\/g, "\\\\").replace(/:/g, "\\:").replace(/'/g, "\\'");
}

