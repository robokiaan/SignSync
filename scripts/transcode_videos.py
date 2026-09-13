"""Re-encode the reference clips for the web: 720p H.264, fast-start, no audio.

The INCLUDE source clips are 1080p at ~19 Mbit/s with the mp4 index (moov
atom) at the END of the file, so a phone has to download the whole clip before
it can show the first frame. Served from a GitHub Release they also came back
as application/octet-stream, which iOS Safari refuses to play at all. The
deployable set therefore lives in the repo (app/static/videos, served by
GitHub Pages as video/mp4 with range support) and is produced from the
originals by this script:

  * 1280x720, H.264 Main profile level 3.1, yuv420p - plays on every phone
  * CRF 23 - ~1.3 Mbit/s, ~0.4 MB per clip, ~110 MB for the whole dictionary
  * -movflags +faststart - index first, so playback starts on the first bytes
  * audio dropped - the app plays every reference muted

Originals are kept, gitignored, in app/static/videos_original/ (and archived
as the "videos-v1" GitHub Release). Timing is preserved exactly (same 25 fps,
same duration), so the hand-labelled hold timestamps in phase_labels.json and
the phases.json built from them stay valid.

ffmpeg comes from the imageio-ffmpeg wheel so nothing needs installing
system-wide:  pip install imageio-ffmpeg

Usage:
    python scripts/transcode_videos.py            # every clip not yet converted
    python scripts/transcode_videos.py --force    # redo all
    python scripts/transcode_videos.py --signs "hello,thank you"
"""
import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "app" / "static" / "videos_original"
OUT_DIR = ROOT / "app" / "static" / "videos"

FFMPEG_ARGS = [
    "-an",
    "-vf", "scale=-2:720",
    "-c:v", "libx264", "-profile:v", "main", "-level", "3.1",
    "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "23",
    "-movflags", "+faststart",
]


def transcode(ffmpeg: str, src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(".part.mp4")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), *FFMPEG_ARGS, str(tmp)]
    subprocess.run(cmd, check=True)
    tmp.replace(dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-encode clips that already exist in the output folder")
    ap.add_argument("--signs", type=str, default=None, help="comma-separated sign names to convert instead of all")
    args = ap.parse_args()

    if not SRC_DIR.exists():
        print(f"missing {SRC_DIR} - move the original 1080p clips there first", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    sources = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() == ".mp4")
    if args.signs:
        wanted = {s.strip().lower() for s in args.signs.split(",") if s.strip()}
        sources = [p for p in sources if p.stem.lower() in wanted]

    done = skipped = 0
    for i, src in enumerate(sources, 1):
        dst = OUT_DIR / (src.stem.lower() + ".mp4")
        if dst.exists() and not args.force:
            skipped += 1
            continue
        transcode(ffmpeg, src, dst)
        done += 1
        print(f"[{i}/{len(sources)}] {src.stem}: {dst.stat().st_size // 1024} KB")

    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.mp4"))
    print(f"\nconverted {done}, skipped {skipped}; output set: {len(list(OUT_DIR.glob('*.mp4')))} clips, {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
