import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import engine, Base, SessionLocal
from app.models import SignDictionary, Lesson, LessonItem

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
        videos_dir = os.path.join(os.path.dirname(__file__), "static", "videos")
        print(f"Loading video files from: {videos_dir}")
        video_files = []
        if os.path.exists(videos_dir):
            video_files = [f[:-4] for f in os.listdir(videos_dir) if f.lower().endswith(".mp4")]
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

    finally:
        db.close()


if __name__ == "__main__":
    seed_db()
