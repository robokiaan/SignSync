import os
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session, selectinload
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.database import get_db, Base, engine, SessionLocal
from app.models import SignDictionary, Lesson, LessonItem, Sentence, SentenceGlossItem
from app import schemas

app = FastAPI(
    title="SignSync ISL API",
    description="Backend API for the SignSync Indian Sign Language learning platform",
    version="1.0.0"
)


def _migrate_sentence_slugs():
    """Self-healing migration for the `slug` column added to `sentences` after
    some hosts already had rows: adds the column if missing (SQLite ALTER),
    then backfills any NULL slug from that row's own gloss so existing DBs
    don't need a manual migration step or a reseed."""
    inspector = inspect(engine)
    if "sentences" not in inspector.get_table_names():
        return
    columns = {c["name"] for c in inspector.get_columns("sentences")}
    if "slug" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE sentences ADD COLUMN slug VARCHAR(200)"))

    from app.seed import slugify_gloss

    db = SessionLocal()
    try:
        missing = (
            db.query(Sentence)
            .options(selectinload(Sentence.items).selectinload(SentenceGlossItem.sign))
            .filter((Sentence.slug.is_(None)) | (Sentence.slug == ""))
            .all()
        )
        if not missing:
            return
        used = {
            row[0] for row in db.query(Sentence.slug).filter(Sentence.slug.isnot(None)).all() if row[0]
        }
        for sentence in missing:
            words = sorted(sentence.items, key=lambda item: item.sort_order)
            base = slugify_gloss(item.sign.sign_name for item in words) or f"sentence-{sentence.id}"
            slug, n = base, 2
            while slug in used:
                slug = f"{base}-{n}"
                n += 1
            sentence.slug = slug
            used.add(slug)
        db.commit()
    finally:
        db.close()


@app.on_event("startup")
def seed_if_empty():
    """Ensure tables exist and the dictionary is populated. Lets the app boot
    with data on a fresh host (where the DB is not committed and the filesystem
    may be ephemeral) without a separate seed step."""
    Base.metadata.create_all(bind=engine)
    _migrate_sentence_slugs()
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


@app.get("/api/sentences/by-slug/{slug}", response_model=schemas.SentenceResponse)
def get_sentence_by_slug(slug: str, db: Session = Depends(get_db)):
    sentence = (
        db.query(Sentence)
        .options(selectinload(Sentence.items).selectinload(SentenceGlossItem.sign))
        .filter(Sentence.slug == slug)
        .first()
    )
    if not sentence:
        raise HTTPException(status_code=404, detail="Sentence not found.")
    sentence.items.sort(key=lambda x: x.sort_order)
    return sentence


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


# Serving the static frontend SPA files (mounted last so it doesn't shadow /api routes)
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)


# SPA deep-link fallback: client routes like /dashboard/alright or
# /sentences/i_happy have no matching static file, so StaticFiles 404s them.
# Serve index.html instead and let app.js's router take it from there. Scoped
# to real page navigations (Accept: text/html) so a missing asset - e.g. a
# reference video 404 that <video onerror> depends on - still 404s normally
# instead of silently getting HTML back.
@app.exception_handler(StarletteHTTPException)
async def spa_fallback(request, exc):
    if (
        exc.status_code == 404
        and request.method == "GET"
        and not request.url.path.startswith("/api")
        and "text/html" in request.headers.get("accept", "")
    ):
        # no-cache: without it, a browser that heuristically cached this
        # response for one deep-linked path would keep re-serving it (with
        # whatever asset versions were current then) instead of hitting the
        # server again after a deploy ships new app.js/styles.css.
        return FileResponse(os.path.join(static_dir, "index.html"), headers={"Cache-Control": "no-cache"})
    # Not our fallback case (API 404, non-GET, asset 404, etc.) - defer to
    # FastAPI's normal JSON error handling. Re-raising exc here would NOT do
    # that: Starlette doesn't re-dispatch an exception through the handler
    # that just raised it, so it would surface as an unhandled 500 instead.
    return await http_exception_handler(request, exc)


# Explicit route for "/" so it gets the same no-cache treatment as the SPA
# fallback above - StaticFiles' own html=True auto-serve of index.html sets
# no Cache-Control at all, which left it to browser heuristic caching. That's
# exactly what caused app.js/styles.css edits to keep silently serving stale
# during this session's testing: an old cached index.html referencing old
# ?v=NN asset URLs is itself never invalidated, and each ?v=NN URL is cached
# forever once fetched once. Declared before the mount so it takes priority.
@app.get("/")
async def index():
    return FileResponse(os.path.join(static_dir, "index.html"), headers={"Cache-Control": "no-cache"})


app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
