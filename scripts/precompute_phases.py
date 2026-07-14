"""Batch-precompute reference phase models for every sign in the dictionary.

Drives real headless Chromium tabs against the app's own coach.js (the exact
code path the browser uses at prime time), so results are byte-identical to
live in-browser priming. Writes app/static/phases.json, which coach.js loads
at startup to skip live priming entirely for any sign already in the file.

One-time / re-run-after-video-changes build step, not part of the deployed
server. Requires: pip install playwright && playwright install chromium

Usage:
    python scripts/precompute_phases.py [--limit N] [--signs a,b,c] [--concurrency N]
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
PORT = 8010
BASE_URL = f"http://127.0.0.1:{PORT}"
OUT_PATH = ROOT / "app" / "static" / "phases.json"
IDLE_TIMEOUT_S = 20        # give up if no sampling progress for this long
ABS_TIMEOUT_S = 180        # hard cap per sign regardless of progress
POLL_INTERVAL_S = 0.3


def wait_for_server(timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(BASE_URL, timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def fetch_sign_names():
    with urllib.request.urlopen(f"{BASE_URL}/api/dictionary") as r:
        data = json.load(r)
    return [s["sign_name"] for s in data]


async def prime_sign(page, sign_name: str):
    # Waits for actual sampling PROGRESS rather than a fixed deadline: headless
    # per-frame inference speed varies a lot (software WebGL readback), so a
    # fixed timeout either wastes time or - worse - fires while priming is
    # still genuinely in-flight. Moving to the next sign while an old prime is
    # still running races coach.js's shared `refSequence` buffer and corrupts
    # data, so only bail once progress has truly stalled.
    await page.evaluate("(name) => startPractice(name, 'precompute')", sign_name)
    key = sign_name.lower()
    start = time.time()
    last_progress_t = start
    last_len = -1
    while True:
        now = time.time()
        if now - start > ABS_TIMEOUT_S:
            return None
        state = await page.evaluate(
            "(k) => ({ model: phaseCache[k] || null, refLen: refSequence.length })", key
        )
        if state["model"]:
            return state["model"]
        if state["refLen"] != last_len:
            last_len = state["refLen"]
            last_progress_t = now
        elif now - last_progress_t > IDLE_TIMEOUT_S:
            return None
        await asyncio.sleep(POLL_INTERVAL_S)


async def worker(browser, worker_id, signs, results, errors):
    context = await browser.new_context()
    page = await context.new_page()
    await page.goto(BASE_URL)
    await page.wait_for_load_state("networkidle")

    for i, sign_name in enumerate(signs):
        t0 = time.time()
        try:
            model = await prime_sign(page, sign_name)
        except Exception as e:
            model = None
            print(f"[w{worker_id}] {sign_name}: EXCEPTION {e}")
        if model:
            results[sign_name.lower()] = model
            holds = len(model.get("holds", []))
            print(f"[w{worker_id}] {i + 1}/{len(signs)} {sign_name}: {holds} hold(s) in {time.time() - t0:.1f}s")
        else:
            errors.append(sign_name)
            print(f"[w{worker_id}] {i + 1}/{len(signs)} {sign_name}: FAILED (timed out)")

    await context.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process the first N signs (testing)")
    ap.add_argument("--signs", type=str, default=None, help="comma-separated sign names to process instead of all")
    ap.add_argument("--concurrency", type=int, default=1, help="parallel browser tabs (keep at 1 - see prime_sign)")
    args = ap.parse_args()

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
        cwd=str(ROOT),
    )
    results = {}
    errors = []
    try:
        if not wait_for_server():
            raise RuntimeError("dev server did not come up on port %d" % PORT)

        if args.signs:
            sign_names = [s.strip() for s in args.signs.split(",") if s.strip()]
        else:
            sign_names = fetch_sign_names()
            if args.limit:
                sign_names = sign_names[: args.limit]
        print(f"Precomputing phase models for {len(sign_names)} sign(s)...")

        # Merge with any existing phases.json so a partial/limited run doesn't
        # clobber previously computed signs.
        if OUT_PATH.exists():
            try:
                results = json.loads(OUT_PATH.read_text())
            except Exception:
                results = {}

        concurrency = max(1, min(args.concurrency, len(sign_names)))
        chunks = [sign_names[i::concurrency] for i in range(concurrency)]

        async with async_playwright() as p:
            # Headless Chromium defaults to software-rendered WebGL here, which
            # makes MediaPipe's GL texture readback the bottleneck (~1 frame/s).
            # Forcing ANGLE/D3D11 GPU rendering cuts that to ~5-6 frames/s after
            # a one-time ~35s per-page warmup (WASM init + shader compile).
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

    OUT_PATH.write_text(json.dumps(results, separators=(",", ":")))
    print(f"\nWrote {len(results)} phase model(s) to {OUT_PATH}")
    if errors:
        print(f"Failed ({len(errors)}): {errors}")


if __name__ == "__main__":
    asyncio.run(main())
