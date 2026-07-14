"""Build app/static/phases.json from manually-labeled hold timestamps.

Reads scripts/phase_labels.json (sign -> [hold timestamps in seconds], picked
by visually reviewing scripts/contact_sheets/*.jpg), drives the app's own
coach.js in headless Chromium to sample each reference video into a feature
sequence exactly like live priming does, then builds each sign's phase model
directly from the manual timestamps instead of coach.js's auto-segmentation
(buildPhaseModel's motion-based hold detection, which produced 1-phase models
for continuous-motion signs like "summer" that have no low-motion pause).

The averaging/moveDirs/requires logic is copied from coach.js's
buildPhaseModel/avgFrames/positionDelta/requiredLimbs so the output shape is
byte-compatible with what the app already consumes.

IMPORTANT: app/static/phases.json must be empty/absent while this runs, or
primeReference() will short-circuit on the existing entry and never populate
refSequence for that sign. This script moves the current file aside for the
duration of the run and restores+overwrites it at the end.

Usage:
    python scripts/build_phases_from_labels.py [--signs a,b,c] [--limit N]
"""
import argparse
import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
PORT = 8011
BASE_URL = f"http://127.0.0.1:{PORT}"
LABELS_PATH = ROOT / "scripts" / "phase_labels.json"
OUT_PATH = ROOT / "app" / "static" / "phases.json"
IDLE_TIMEOUT_S = 20
ABS_TIMEOUT_S = 180
POLL_INTERVAL_S = 0.3

BUILD_JS = """
(timestamps) => {
    const seq = refSequence;
    const n = seq.length;
    if (n === 0) return null;
    const dt = 1 / SAMPLE_FPS;
    const idxFor = (t) => Math.min(Math.max(Math.round(t / dt), 0), n - 1);
    const holds = timestamps.map((t) => {
        const idx = idxFor(t);
        const lo = Math.max(0, idx - 1), hi = Math.min(n - 1, idx + 1);
        return avgFrames(seq.slice(lo, hi + 1));
    });
    const moveDirs = holds.map((h, p) => {
        if (p === 0) return null;
        const dir = positionDelta(holds[p - 1].features, holds[p].features);
        const mag = Math.sqrt(dir.reduce((s, v) => s + v * v, 0));
        if (mag < 0.08) return null;
        return dir.map((v) => v / mag);
    });
    const requires = requiredLimbs(seq, holds);
    return { holds, moveDirs, holdTimes: timestamps.slice(), requires };
}
"""


def wait_for_server(timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE_URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


async def prime_and_build(page, sign_name: str, timestamps: list):
    await page.evaluate("(name) => startPractice(name, 'manual-build')", sign_name)
    key = sign_name.lower()
    start = time.time()
    last_progress_t = start
    last_len = -1
    while True:
        now = time.time()
        if now - start > ABS_TIMEOUT_S:
            return None
        state = await page.evaluate(
            "(k) => ({ ready: !!phaseCache[k], refLen: refSequence.length })", key
        )
        if state["ready"]:
            break
        if state["refLen"] != last_len:
            last_len = state["refLen"]
            last_progress_t = now
        elif now - last_progress_t > IDLE_TIMEOUT_S:
            return None
        await asyncio.sleep(POLL_INTERVAL_S)

    return await page.evaluate(BUILD_JS, timestamps)


async def worker(browser, worker_id, items, results, errors):
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(BASE_URL)
    await page.wait_for_load_state("networkidle")

    for i, (sign_name, timestamps) in enumerate(items):
        t0 = time.time()
        try:
            model = await prime_and_build(page, sign_name, timestamps)
        except Exception as e:
            model = None
            print(f"[w{worker_id}] {sign_name}: EXCEPTION {e}")
        if model:
            results[sign_name.lower()] = model
            holds = len(model.get("holds", []))
            print(f"[w{worker_id}] {i + 1}/{len(items)} {sign_name}: {holds} hold(s) in {time.time() - t0:.1f}s")
        else:
            errors.append(sign_name)
            print(f"[w{worker_id}] {i + 1}/{len(items)} {sign_name}: FAILED (timed out)")

    await context.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--signs", type=str, default=None)
    ap.add_argument("--concurrency", type=int, default=1, help="parallel browser tabs (keep at 1 - shared refSequence)")
    args = ap.parse_args()

    labels = json.loads(LABELS_PATH.read_text())
    if args.signs:
        wanted = [s.strip() for s in args.signs.split(",") if s.strip()]
        items = [(s, labels[s]) for s in wanted]
    else:
        items = list(labels.items())
        if args.limit:
            items = items[: args.limit]

    backup_path = OUT_PATH.with_suffix(".json.bak")
    had_existing = OUT_PATH.exists()
    if had_existing:
        OUT_PATH.replace(backup_path)
    OUT_PATH.write_text("{}")  # empty, so primeReference() always live-primes

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=str(ROOT),
    )
    results = {}
    errors = []
    try:
        if not wait_for_server():
            raise RuntimeError("dev server did not come up on port %d" % PORT)

        print(f"Building phase models for {len(items)} sign(s) from manual labels...")

        concurrency = max(1, min(args.concurrency, len(items)))
        chunks = [items[i::concurrency] for i in range(concurrency)]

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=[
                "--use-gl=angle", "--use-angle=d3d11", "--ignore-gpu-blocklist",
                "--enable-gpu-rasterization", "--enable-zero-copy",
            ])
            await asyncio.gather(
                *(worker(browser, wid, chunk, results, errors) for wid, chunk in enumerate(chunks) if chunk)
            )
            await browser.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    # Full rebuild: if this was a partial/limited run, merge onto the backup
    # (previous full result) rather than the fetched-empty placeholder.
    merged = {}
    if had_existing and (args.signs or args.limit):
        try:
            merged = json.loads(backup_path.read_text())
        except Exception:
            merged = {}
    merged.update(results)
    OUT_PATH.write_text(json.dumps(merged, separators=(",", ":")))
    print(f"\nWrote {len(merged)} phase model(s) to {OUT_PATH}")
    if errors:
        print(f"Failed ({len(errors)}): {errors}")


if __name__ == "__main__":
    asyncio.run(main())
