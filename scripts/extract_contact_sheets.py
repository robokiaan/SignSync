"""Generate one labeled contact-sheet image per sign for manual hold labeling.

Samples each reference video at a fixed time interval (finer than the old
motion-based auto-segmentation's 5fps, so brief holds aren't missed), tiles
the frames into a grid with a timestamp caption on each cell, and writes one
JPEG per sign to scripts/contact_sheets/. A human (or Claude, visually)
reviews each sheet and records which timestamps are genuine holds.

Usage:
    python scripts/extract_contact_sheets.py [--signs a,b,c] [--interval 0.15]
"""
import argparse
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "app" / "static" / "videos"
OUT_DIR = ROOT / "scripts" / "contact_sheets"

THUMB_W, THUMB_H = 220, 124
COLS = 5
CAPTION_H = 20
PAD = 4


def extract_frames(video_path, interval_s):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0

    frames = []
    t = 0.0
    while t < duration - 1e-3 or not frames:
        idx = min(int(round(t * fps)), max(frame_count - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((t, frame))
        if frame_count <= 1:
            break
        t += interval_s
    cap.release()
    return frames, duration


def make_contact_sheet(frames, out_path):
    rows = math.ceil(len(frames) / COLS)
    sheet_w = COLS * (THUMB_W + PAD) + PAD
    sheet_h = rows * (THUMB_H + CAPTION_H + PAD) + PAD
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for i, (t, frame) in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((THUMB_W, THUMB_H))
        col, row = i % COLS, i // COLS
        x = PAD + col * (THUMB_W + PAD)
        y = PAD + row * (THUMB_H + CAPTION_H + PAD)
        # Center the (aspect-preserved) thumbnail in its cell.
        ox = x + (THUMB_W - img.width) // 2
        oy = y + (THUMB_H - img.height) // 2
        sheet.paste(img, (ox, oy))
        draw.rectangle([x, y, x + THUMB_W, y + THUMB_H], outline=(90, 90, 90))
        caption = f"#{i} t={t:.2f}s"
        draw.text((x + 2, y + THUMB_H + 2), caption, fill=(255, 255, 0), font=font)

    sheet.save(out_path, quality=85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", type=str, default=None, help="comma-separated sign names (filenames minus .mp4)")
    ap.add_argument("--interval", type=float, default=0.15, help="seconds between sampled frames")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.signs:
        names = [s.strip() for s in args.signs.split(",") if s.strip()]
        video_paths = [VIDEOS_DIR / f"{n}.mp4" for n in names]
    else:
        video_paths = sorted(VIDEOS_DIR.glob("*.mp4"))

    manifest = {}
    for vp in video_paths:
        if not vp.exists():
            print(f"MISSING: {vp}")
            continue
        sign = vp.stem
        frames, duration = extract_frames(vp, args.interval)
        out_path = OUT_DIR / f"{sign}.jpg"
        make_contact_sheet(frames, out_path)
        manifest[sign] = {"duration": round(duration, 3), "frame_count": len(frames)}
        print(f"{sign}: {len(frames)} frames, {duration:.2f}s -> {out_path.name}")

    print(f"\nDone: {len(manifest)} contact sheets written to {OUT_DIR}")


if __name__ == "__main__":
    main()
