import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base, SessionLocal
from app.models import SignDictionary, Lesson, LessonItem, Sentence, SentenceGlossItem

# Maps each sign to a curriculum category. Signs not listed fall back to "General".
CATEGORIES_MAP = {
    "colours": ["blue", "black", "brown", "green", "grey", "orange", "pink", "red", "white", "yellow", "colour"],
    "seasons": ["summer", "spring", "winter", "fall", "season", "ex. monsoon", "monsoon"],
    "days and time": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "tomorrow", "yesterday", "week", "month", "year", "time", "morning", "afternoon", "evening", "night", "second", "hour", "minute"],
    "animals": ["animal", "cat", "dog", "cow", "horse", "bird", "fish", "mouse"],
    "clothes": ["clothing", "dress", "hat", "pant", "pocket", "shirt", "shoes", "skirt", "suit", "t-shirt"],
    "pronouns": ["i", "you", "he", "she", "it", "we", "they", "you (plural)"],
    "greetings": ["hello", "good morning", "good afternoon", "good evening", "good night", "thank you", "pleased", "how are you", "alright"],
    "places": ["city", "house", "street or road", "train station", "restaurant", "court", "school", "office", "university", "park", "store or shop", "library", "hospital", "temple", "market", "india", "ground", "bank", "location"],
    "jobs": ["teacher", "student", "lawyer", "doctor", "patient", "waiter", "secretary", "priest", "police", "soldier", "artist", "author", "manager", "reporter", "actor", "job"],
    "transportation": ["plane", "car", "truck", "bicycle", "train", "bus", "boat", "train ticket", "transportation"],
    "home": ["table", "chair", "bed", "dream", "window", "door", "bedroom", "kitchen", "bathroom", "pencil", "pen", "photograph", "soap", "book", "page", "key", "paint", "letter", "paper", "lock", "telephone", "bag", "box", "gift", "card", "ring", "tool"],
    "people": ["son", "daughter", "mother", "father", "parent", "baby", "man", "woman", "brother", "sister", "family", "grandfather", "grandmother", "husband", "wife", "king", "queen", "president", "neighbour", "boy", "girl", "child", "adult", "friend", "player", "crowd"],
    "adjectives": ["loud", "quiet", "happy", "sad", "beautiful", "ugly", "deaf", "blind", "mean", "rich", "poor", "thick", "thin", "expensive", "cheap", "flat", "curved", "male", "female", "tight", "loose", "nice", "high", "low", "soft", "hard", "deep", "shallow", "clean", "dirty", "strong", "weak", "dead", "alive", "heavy", "light", "famous", "long", "short", "tall", "wide", "narrow", "big large", "small little", "slow", "fast", "hot", "cold", "warm", "cool", "new", "bad", "good", "healthy", "sick", "wet", "dry", "old", "young"],
    "society": ["religion", "death", "medicine", "money", "bill", "marriage", "team", "race (ethnicity)", "energy", "war", "peace", "attack", "election", "newspaper", "gun", "technology", "sport", "exercise", "ball", "price", "sign", "science", "god"],
}

BEGINNER_CATEGORIES = {"Greetings", "Pronouns", "Colours"}


def slugify_gloss(sign_names):
    """URL slug for a sentence, e.g. ["i", "happy"] -> "i_happy"."""
    return "_".join(name.strip().lower().replace(" ", "-") for name in sign_names)


def get_category_for_sign(name):
    lname = name.lower()
    for cat, words in CATEGORIES_MAP.items():
        if lname in words:
            return cat.title()
    return "General"


def seed_db():
    print("Re-creating all tables in the SQLite database...")
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Sign list: prefer the local video files; fall back to the committed
        # manifest (app/signs.json) so the app can seed on a host where the large
        # video folder isn't deployed.
        videos_dir = os.path.join(os.path.dirname(__file__), "static", "videos")
        video_files = []
        if os.path.exists(videos_dir):
            video_files = [f[:-4] for f in os.listdir(videos_dir) if f.lower().endswith(".mp4")]
        if not video_files:
            manifest = os.path.join(os.path.dirname(__file__), "signs.json")
            if os.path.exists(manifest):
                import json
                with open(manifest, "r", encoding="utf-8") as f:
                    video_files = json.load(f)
                print(f"No local videos; seeding from manifest {manifest}")
        print(f"Found {len(video_files)} signs to seed.")

        # Seed Sign Dictionary
        db_signs = []
        for name in video_files:
            cat = get_category_for_sign(name)
            desc = f"Practice the sign for '{name.title()}' in the {cat} category."
            sign = SignDictionary(
                sign_name=name.lower(),
                category=cat,
                description=desc,
            )
            db.add(sign)
            db_signs.append(sign)

        db.commit()
        for s in db_signs:
            db.refresh(s)
        print(f"Successfully seeded {len(db_signs)} signs into the dictionary.")

        # Group signs by category and build one lesson per category
        signs_by_cat = {}
        for s in db_signs:
            signs_by_cat.setdefault(s.category, []).append(s)

        for cat, signs in signs_by_cat.items():
            lesson = Lesson(
                title=f"Mastering {cat}",
                category=cat,
                difficulty_level="beginner" if cat in BEGINNER_CATEGORIES else "intermediate",
                description=f"Learn and master Indian Sign Language signs related to {cat}.",
            )
            db.add(lesson)
            db.commit()
            db.refresh(lesson)

            for idx, s in enumerate(sorted(signs, key=lambda x: x.sign_name)):
                db.add(LessonItem(lesson_id=lesson.id, sign_id=s.id, sort_order=idx + 1))
            db.commit()

        print(f"Successfully seeded {len(signs_by_cat)} lessons and curriculum items dynamically.")

        seed_sentences(db, {s.sign_name: s for s in db_signs})

    finally:
        db.close()


def seed_sentences(db, signs_by_name):
    """Seed the curated Sentences practice content (app/sentences.json), each a
    {english, difficulty, gloss} entry where every gloss word must already be a
    seeded SignDictionary entry (case-insensitive). Words not in the dictionary
    are skipped with a warning rather than raising, so a typo in the manifest
    can't take down startup seeding."""
    manifest = os.path.join(os.path.dirname(__file__), "sentences.json")
    if not os.path.exists(manifest):
        return
    import json
    with open(manifest, "r", encoding="utf-8") as f:
        entries = json.load(f)

    count = 0
    used_slugs = set()
    for entry in entries:
        gloss_signs = []
        for word in entry["gloss"]:
            sign = signs_by_name.get(word.lower())
            if not sign:
                print(f"WARNING: sentence '{entry['english']}' references unknown sign '{word}' - skipping sentence.")
                gloss_signs = None
                break
            gloss_signs.append(sign)
        if not gloss_signs:
            continue

        base_slug = slugify_gloss(s.sign_name for s in gloss_signs)
        slug, n = base_slug, 2
        while slug in used_slugs:
            slug = f"{base_slug}-{n}"
            n += 1
        used_slugs.add(slug)

        sentence = Sentence(
            english_text=entry["english"],
            difficulty_level=entry["difficulty"],
            category=entry.get("category"),
            slug=slug,
        )
        db.add(sentence)
        db.commit()
        db.refresh(sentence)

        for idx, sign in enumerate(gloss_signs):
            db.add(SentenceGlossItem(sentence_id=sentence.id, sign_id=sign.id, sort_order=idx + 1))
        db.commit()
        count += 1

    print(f"Successfully seeded {count} sentences.")


if __name__ == "__main__":
    seed_db()
