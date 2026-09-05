"""Report data-quality problems in app/static/phases.json.

The coach grades a learner by comparing their pose against a handful of held
target poses per sign. Those targets are only as good as the reference tracking
behind them, and three specific defects make a sign score wrongly no matter what
the scoring code does:

  1. Duplicate checkpoints - two holds in one sign that came out identical.
     Whichever is checked second can never be the better match, so the sign
     cannot be completed.
  2. Hand-less checkpoints - the hand tracker lost the hands on the reference,
     so the target carries no handshape. Any handshape then scores full marks.
     Only matters where the hand was RAISED; a hand resting at the signer's side
     carries no meaning whether it was tracked or not.
  3. Rest-only checkpoints - a hold marked while the hands are down, i.e. in the
     idle part of the clip rather than on the sign itself. A labelling mistake.

Run before and after any change to SAMPLE_FPS or a phases.json rebuild:

    python scripts/check_phase_health.py [--json path/to/phases.json]

Exits non-zero if any sign is uncompletable, so it can gate a rebuild.
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mirrors coach.js: FEATURE_GROUPS, FEATURE_WEIGHTS, HOLD_MATCH_THRESHOLD,
# HAND_ACTIVE_Y. Kept in sync by hand - this script is a check, not a dependency.
FEATURE_GROUPS = [
    dict(start=0, end=9, needRight=True, needLeft=False, needPose=False),
    dict(start=9, end=18, needRight=False, needLeft=True, needPose=False),
    dict(start=18, end=22, needRight=False, needLeft=False, needPose=True),
    dict(start=22, end=24, needRight=False, needLeft=False, needPose=True),
    dict(start=24, end=26, needRight=False, needLeft=False, needPose=True),
]
WEIGHTS = [1, 1, 1, 1, 1, .4, .4, .2, .5,
           1, 1, 1, 1, 1, .4, .4, .2, .5,
           1, 1, .8, .8, .6, .6, .6, .6]
WEIGHT_TOTAL = sum(WEIGHTS)
HOLD_MATCH_THRESHOLD = 0.50
HAND_ACTIVE_Y = 0.85
DUPLICATE_Q = 0.98   # two holds this close are effectively the same pose


def clamp01(v):
    return 0.0 if v < 0 else (1.0 if v > 1 else v)


def distance(a, b):
    """Weighted distance between two reference holds, over what both contain."""
    av, bv = a["visibility"], b["visibility"]
    af, bf = a["features"], b["features"]
    total = 0.0
    for g in FEATURE_GROUPS:
        ok = ((not g["needRight"] or (av["rightHand"] and bv["rightHand"]))
              and (not g["needLeft"] or (av["leftHand"] and bv["leftHand"]))
              and (not g["needPose"] or (av["pose"] and bv["pose"])))
        for i in range(g["start"], g["end"]):
            total += WEIGHTS[i] * ((af[i] - bf[i]) ** 2 if ok else 1.0)
    return math.sqrt(total / WEIGHT_TOTAL)


def quality(a, b):
    return clamp01(1 - distance(a, b) / HOLD_MATCH_THRESHOLD)


def hand_raised(hold, side):
    """Wrist height comes from the pose landmarks, so it survives hand-tracking
    loss. Lower value = higher up."""
    if not hold["visibility"]["pose"]:
        return False
    y = hold["features"][23] if side == "right" else hold["features"][25]
    return y < HAND_ACTIVE_Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=os.path.join(ROOT, "app", "static", "phases.json"))
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        phases = json.load(f)

    duplicates, handless, rest_only = [], [], []

    for name, model in sorted(phases.items()):
        holds = model.get("holds", [])

        for p, hold in enumerate(holds):
            vis = hold["visibility"]
            if not vis["rightHand"] and not vis["leftHand"]:
                raised = hand_raised(hold, "right") or hand_raised(hold, "left")
                handless.append((name, p + 1, len(holds), raised))
                if not raised and len(holds) == 1:
                    # A one-hold sign whose only pose has the hands down was
                    # marked in the idle part of the clip, not on the sign.
                    rest_only.append((name, p + 1))

        for p in range(len(holds)):
            for q_i in range(p + 1, len(holds)):
                if quality(holds[p], holds[q_i]) >= DUPLICATE_Q:
                    duplicates.append((name, p + 1, q_i + 1, len(holds)))

    total_signs = len(phases)
    print("phases.json: %s" % args.json)
    print("signs: %d" % total_signs)
    print()

    print("1. UNCOMPLETABLE - duplicate checkpoints (the later one can never win)")
    if duplicates:
        for name, a, b, n in duplicates:
            print("     %-18s checkpoints %d and %d of %d are identical" % (name, a, b, n))
        print("   affected signs: %d" % len({d[0] for d in duplicates}))
    else:
        print("     none")
    print()

    raised = [h for h in handless if h[3]]
    resting = [h for h in handless if not h[3]]
    print("2. HAND-LESS CHECKPOINTS - reference carries no handshape")
    print("     total: %d across %d signs" % (len(handless), len({h[0] for h in handless})))
    print("     hand RAISED (real gap, needs re-marking): %d across %d signs"
          % (len(raised), len({h[0] for h in raised})))
    print("     hand RESTING (excused, no handshape to grade): %d" % len(resting))
    if raised:
        print("     re-mark:", ", ".join(sorted({h[0] for h in raised})))
    print()

    print("3. REST-ONLY - single-checkpoint signs marked while the hands are down")
    if rest_only:
        for name, p in rest_only:
            print("     %-18s only checkpoint is in the idle part of the clip" % name)
    else:
        print("     none")
    print()

    hand_counts = {}
    for name, model in phases.items():
        need = model.get("requires", {}).get("hands")
        hand_counts[name] = need
    from collections import Counter
    print("hand requirement spread: %s" % dict(sorted(Counter(hand_counts.values()).items(),
                                                      key=lambda kv: (kv[0] is None, kv[0]))))

    return 1 if duplicates else 0


if __name__ == "__main__":
    sys.exit(main())
