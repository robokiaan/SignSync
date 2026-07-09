import os
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session, selectinload

from app.database import get_db
from app.models import SignDictionary, Lesson, LessonItem
from app import schemas

app = FastAPI(
    title="SignSync ISL API",
    description="Backend API for the SignSync Indian Sign Language learning platform",
    version="1.0.0"
)


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
