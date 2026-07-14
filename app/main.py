import os
import random
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session, selectinload

from app.database import get_db, Base, engine, SessionLocal
from app.gloss_matching import build_alias_index, parse_sentence
from app.models import SignDictionary, Lesson, LessonItem, Sentence, SentenceGlossItem
from app import schemas

app = FastAPI(
    title="SignSync ISL API",
    description="Backend API for the SignSync Indian Sign Language learning platform",
    version="1.0.0"
)


@app.on_event("startup")
def seed_if_empty():
    """Ensure tables exist and the dictionary is populated. Lets the app boot
    with data on a fresh host (where the DB is not committed and the filesystem
    may be ephemeral) without a separate seed step."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Also reseed if Sentence is empty: on an existing DB where the
        # dictionary is already populated but this table was just added,
        # only checking SignDictionary would leave it empty forever.
        if db.query(SignDictionary).count() == 0 or db.query(Sentence).count() == 0:
            from app.seed import seed_db
            seed_db()
    finally:
        db.close()


# LESSONS ROUTE: Get All
@app.get("/api/lessons", response_model=List[schemas.LessonResponse])
def get_lessons(db: Session = Depends(get_db)):
    # Eager-load items and their signs to avoid N+1 queries, ordered by sort_order.
    lessons = (
        db.query(Lesson)
        .options(selectinload(Lesson.items).selectinload(LessonItem.sign))
        .all()
    )
    for lesson in lessons:
        lesson.items.sort(key=lambda x: x.sort_order)
    return lessons


# DICTIONARY ROUTES
@app.get("/api/dictionary", response_model=List[schemas.SignResponse])
def get_dictionary(category: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(SignDictionary)
    if category:
        query = query.filter(SignDictionary.category == category)
    return query.all()


@app.get("/api/dictionary/{sign_name}", response_model=schemas.SignResponse)
def get_sign(sign_name: str, db: Session = Depends(get_db)):
    sign = db.query(SignDictionary).filter(SignDictionary.sign_name == sign_name.lower()).first()
    if not sign:
        raise HTTPException(status_code=404, detail="Sign not found in dictionary.")
    return sign


# SENTENCES ROUTES
@app.get("/api/sentences", response_model=List[schemas.SentenceResponse])
def get_sentences(db: Session = Depends(get_db)):
    sentences = (
        db.query(Sentence)
        .options(selectinload(Sentence.items).selectinload(SentenceGlossItem.sign))
        .all()
    )
    for sentence in sentences:
        sentence.items.sort(key=lambda x: x.sort_order)
    return sentences


# Lazily-built cache: {tuple(tokens): sign_name}, from build_alias_index() over
# every SignDictionary row. Dictionary content is static after seeding (no
# runtime write endpoints), so a process-lifetime cache is safe.
_alias_index_cache = None


def get_alias_index(db: Session):
    global _alias_index_cache
    if _alias_index_cache is None:
        names = [s.sign_name for s in db.query(SignDictionary).all()]
        _alias_index_cache = build_alias_index(names)
    return _alias_index_cache


# NOTE: declared BEFORE /api/sentences/{sentence_id} - FastAPI/Starlette
# matches routes by declaration order, and {sentence_id} would otherwise
# swallow "/api/sentences/parse" and "/api/sentences/generate" as a literal
# path segment (then 422 on the int type check) before ever reaching these.
@app.post("/api/sentences/parse", response_model=schemas.ParseSentenceResponse)
def parse_sentence_route(payload: schemas.ParseSentenceRequest, db: Session = Depends(get_db)):
    index = get_alias_index(db)
    gloss, unmatched = parse_sentence(payload.text, index)
    return {"gloss": gloss, "unmatched": unmatched}


# Template-based random sentence synthesis (not a translator, not an LLM call)
# for the "Auto-generate" button: pick a template, fill its two slots with a
# random sign from each named category. English + gloss are produced together
# so there's no parsing ambiguity for generated text.
_PRONOUN_COPULA = {
    "i": "am", "you": "are", "he": "is", "she": "is", "it": "is",
    "we": "are", "they": "are", "you (plural)": "are",
}
_GENERATE_TEMPLATES = [
    ("Pronouns", "Adjectives", lambda a, b: f"{a.title()} {_PRONOUN_COPULA.get(a, 'is')} {b}."),
    ("People", "Adjectives", lambda a, b: f"{a.title()} is {b}."),
    ("Days And Time", "Adjectives", lambda a, b: f"{a.title()} is {b}."),
    ("Animals", "Adjectives", lambda a, b: f"The {a} is {b}."),
    ("Clothes", "Adjectives", lambda a, b: f"The {a} is {b}."),
    ("Transportation", "Adjectives", lambda a, b: f"The {a} is {b}."),
    ("Jobs", "Adjectives", lambda a, b: f"The {a} is {b}."),
    ("Places", "Adjectives", lambda a, b: f"{a.title()} is {b}." if a == "india" else f"The {a} is {b}."),
    # Category order = presentation order = gloss order here too (animal
    # first, colour second), so gloss stays consistent with every other
    # template instead of secretly reversing what "a"/"b" mean.
    ("Animals", "Colours", lambda a, b: f"The {a} is {b}."),
]


@app.get("/api/sentences/generate", response_model=schemas.GenerateSentenceResponse)
def generate_sentence(db: Session = Depends(get_db)):
    cat_a, cat_b, formatter = random.choice(_GENERATE_TEMPLATES)
    sign_a = db.query(SignDictionary).filter(SignDictionary.category == cat_a).order_by(func.random()).first()
    sign_b = db.query(SignDictionary).filter(SignDictionary.category == cat_b).order_by(func.random()).first()
    if not sign_a or not sign_b:
        raise HTTPException(status_code=500, detail="Not enough dictionary signs to generate a sentence.")
    english = formatter(sign_a.sign_name, sign_b.sign_name)
    return {"english": english, "gloss": [sign_a.sign_name, sign_b.sign_name]}


@app.get("/api/sentences/{sentence_id}", response_model=schemas.SentenceResponse)
def get_sentence(sentence_id: int, db: Session = Depends(get_db)):
    sentence = (
        db.query(Sentence)
        .options(selectinload(Sentence.items).selectinload(SentenceGlossItem.sign))
        .filter(Sentence.id == sentence_id)
        .first()
    )
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found.")
    sentence.items.sort(key=lambda x: x.sort_order)
    return sentence


# BROWSER LOGS ROUTE (client-side error reporting during development)
@app.post("/api/log-error")
def log_browser_error(data: dict = Body(...)):
    print(f"\n[!!! BROWSER ERROR REPORT !!!] {data}\n")
    return {"status": "logged"}


# Serving the static frontend SPA files (mounted last so it doesn't shadow /api routes)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
