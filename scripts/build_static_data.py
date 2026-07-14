"""Generate the static JSON data GitHub Pages needs (app/static/data/*.json).

The FastAPI backend (app/main.py) computes this same shape from a live SQLite
DB at request time. Pages has no server, so this bakes the identical output
to disk once, from the same source manifests (app/signs.json, app/sentences.json)
and the same grouping/slug logic (app/seed.py) - no DB involved, deterministic
across runs so diffs stay meaningful.

Usage: python scripts/build_static_data.py
Re-run whenever app/signs.json or app/sentences.json changes; the GitHub
Actions Pages workflow also runs this on every push so it can't go stale.
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.seed import CATEGORIES_MAP, BEGINNER_CATEGORIES, get_category_for_sign, slugify_gloss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "app", "static", "data")


def build_dictionary():
    manifest_path = os.path.join(ROOT, "app", "signs.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        names = json.load(f)

    signs = []
    for i, name in enumerate(sorted(names, key=str.lower), start=1):
        cat = get_category_for_sign(name)
        signs.append({
            "id": i,
            "sign_name": name.lower(),
            "category": cat,
            "description": f"Practice the sign for '{name.title()}' in the {cat} category.",
        })
    return signs


def build_lessons(signs):
    by_name = {s["sign_name"]: s for s in signs}
    by_cat = {}
    for s in signs:
        by_cat.setdefault(s["category"], []).append(s)

    # Category order matches CATEGORIES_MAP's own definition order (not
    # alphabetical - the old DB-backed backend's order was filesystem/manifest
    # dependent and not reliably reproducible, but CATEGORIES_MAP's order is
    # deterministic and was clearly deliberately chosen). "General" (the
    # fallback for unmapped signs, not itself a CATEGORIES_MAP key) goes last.
    cat_order = [cat.title() for cat in CATEGORIES_MAP] + ["General"]
    ordered_cats = sorted(by_cat, key=lambda c: cat_order.index(c) if c in cat_order else len(cat_order))

    lessons = []
    for lesson_id, cat in enumerate(ordered_cats, start=1):
        cat_signs = sorted(by_cat[cat], key=lambda s: s["sign_name"])
        lessons.append({
            "id": lesson_id,
            "title": f"Mastering {cat}",
            "category": cat,
            "difficulty_level": "beginner" if cat in BEGINNER_CATEGORIES else "intermediate",
            "description": f"Learn and master Indian Sign Language signs related to {cat}.",
            "items": [
                {"id": idx, "sign_id": s["id"], "sort_order": idx, "sign": s}
                for idx, s in enumerate(cat_signs, start=1)
            ],
        })
    return lessons


def build_sentences(signs):
    by_name = {s["sign_name"]: s for s in signs}
    manifest_path = os.path.join(ROOT, "app", "sentences.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    sentences = []
    used_slugs = set()
    item_id = 1
    for sentence_id, entry in enumerate(entries, start=1):
        gloss_signs = []
        for word in entry["gloss"]:
            sign = by_name.get(word.lower())
            if not sign:
                print(f"WARNING: sentence '{entry['english']}' references unknown sign '{word}' - skipping sentence.")
                gloss_signs = None
                break
            gloss_signs.append(sign)
        if not gloss_signs:
            continue

        base_slug = slugify_gloss(s["sign_name"] for s in gloss_signs)
        slug, n = base_slug, 2
        while slug in used_slugs:
            slug = f"{base_slug}-{n}"
            n += 1
        used_slugs.add(slug)

        items = []
        for idx, sign in enumerate(gloss_signs, start=1):
            items.append({"id": item_id, "sign_id": sign["id"], "sort_order": idx, "sign": sign})
            item_id += 1

        sentences.append({
            "id": sentence_id,
            "english_text": entry["english"],
            "difficulty_level": entry["difficulty"],
            "category": entry.get("category"),
            "slug": slug,
            "items": items,
        })
    return sentences


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    signs = build_dictionary()
    lessons = build_lessons(signs)
    sentences = build_sentences(signs)

    for name, data in [("dictionary.json", signs), ("lessons.json", lessons), ("sentences.json", sentences)]:
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        print(f"Wrote {name}: {len(data)} entries")


if __name__ == "__main__":
    main()
