import os
import datetime
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import bcrypt

from app.database import get_db, Base, engine
from app.models import User, UserProfile, SignDictionary, Lesson, LessonItem, PracticeSession
from app import schemas

# Security settings
SECRET_KEY = "signsync_super_secret_key_change_me_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days token

app = FastAPI(
    title="SignSync ISL API",
    description="Backend API for the SignSync Indian Sign Language learning platform",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions using modern bcrypt library directly
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Dependency to get current user from token
async def get_current_user(token: str = Depends(lambda: None), db: Session = Depends(get_db)):
    # Look for Authorization header in request (custom extraction because of frontend ease of use)
    # We will write a custom check to support headers or query param tokens
    return None

# Mocked implementation to bypass session storage & gamification writes
async def get_current_user_from_request(request: Request):
    class MockProfile:
        xp = 0
        current_level = 1
        current_streak = 0
        longest_streak = 0
        last_active_date = None
        avatar_url = "https://api.dicebear.com/7.x/bottts/svg"
        
    class MockUser:
        id = "mock-user-id"
        username = "Guest Learner"
        email = "guest@example.com"
        profile = MockProfile()
        
    return MockUser()

# AUTH ROUTE: Register
@app.post("/api/auth/register", response_model=schemas.Token)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if username or email already exists
    db_user = db.query(User).filter((User.username == user_in.username) | (User.email == user_in.email)).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Username or email already registered."
        )
        
    # Create new User
    hashed_password = get_password_hash(user_in.password)
    new_user = User(
        username=user_in.username,
        email=user_in.email,
        password_hash=hashed_password
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Create associated UserProfile
    new_profile = UserProfile(
        user_id=new_user.id,
        xp=0,
        current_level=1,
        current_streak=0,
        longest_streak=0,
        last_active_date=None,
        avatar_url=f"https://api.dicebear.com/7.x/bottts/svg?seed={new_user.username}"
    )
    db.add(new_profile)
    db.commit()
    
    # Create and return access token
    access_token = create_access_token(data={"sub": new_user.id})
    return {"access_token": access_token, "token_type": "bearer"}

# AUTH ROUTE: Login
@app.post("/api/auth/login", response_model=schemas.Token)
def login(login_in: schemas.UserLogin, db: Session = Depends(get_db)):
    # Check by username or email
    user = db.query(User).filter(
        (User.username == login_in.username_or_email) | (User.email == login_in.username_or_email)
    ).first()
    
    if not user or not verify_password(login_in.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Create and return access token
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

# USER ROUTE: Get Profile
@app.get("/api/profile", response_model=schemas.UserProfileResponse)
def get_profile(current_user: User = Depends(get_current_user_from_request), db: Session = Depends(get_db)):
    profile = current_user.profile
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "xp": profile.xp,
        "current_level": profile.current_level,
        "current_streak": profile.current_streak,
        "longest_streak": profile.longest_streak,
        "last_active_date": profile.last_active_date,
        "avatar_url": profile.avatar_url
    }

# USER ROUTE: Update Profile
@app.patch("/api/profile", response_model=schemas.UserProfileResponse)
def update_profile(
    avatar_url: Optional[str] = Body(None),
    current_user: User = Depends(get_current_user_from_request),
    db: Session = Depends(get_db)
):
    profile = current_user.profile
    if avatar_url is not None:
        profile.avatar_url = avatar_url
    db.commit()
    db.refresh(profile)
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "xp": profile.xp,
        "current_level": profile.current_level,
        "current_streak": profile.current_streak,
        "longest_streak": profile.longest_streak,
        "last_active_date": profile.last_active_date,
        "avatar_url": profile.avatar_url
    }



# LESSONS ROUTE: Get All
@app.get("/api/lessons", response_model=List[schemas.LessonResponse])
def get_lessons(db: Session = Depends(get_db)):
    # Eager load items and sign relations
    lessons = db.query(Lesson).all()
    # Sort items internally
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

# PRACTICE LOG ROUTE
@app.post("/api/practice", response_model=schemas.PracticeSessionResponse)
def log_practice(
    session_in: schemas.PracticeSessionCreate,
    current_user: User = Depends(get_current_user_from_request),
    db: Session = Depends(get_db)
):
    sign = db.query(SignDictionary).filter(SignDictionary.id == session_in.sign_id).first()
    if not sign:
        raise HTTPException(status_code=404, detail="Sign not found.")
        
    today = datetime.date.today()
    xp_gained = 0
    streak_bonus = 0
    
    # Only award XP/streaks if score is at least 50% (50 points out of 100)
    # [GAMIFICATION DISABLED]
    # if session_in.score >= 50:
    #     start_of_today = datetime.datetime.combine(today, datetime.time.min)
    #     end_of_today = datetime.datetime.combine(today, datetime.time.max)
    #     
    #     # Check if they have already successfully practiced this sign today
    #     already_practiced_today = db.query(PracticeSession).filter(
    #         PracticeSession.user_id == current_user.id,
    #         PracticeSession.sign_id == session_in.sign_id,
    #         PracticeSession.score >= 50,
    #         PracticeSession.completed_at >= start_of_today,
    #         PracticeSession.completed_at <= end_of_today
    #     ).first()
    #     
    #     if not already_practiced_today:
    #         # Base XP: proportional to score, plus bonus for high scores (>= 80)
    #         xp_gained = int(session_in.score * 0.5)
    #         if session_in.score >= 80:
    #             xp_gained += 15  # high score bonus

    # [DATABASE WRITE DISABLED]
    # new_session = PracticeSession(
    #     user_id=current_user.id,
    #     sign_id=session_in.sign_id,
    #     score=session_in.score
    # )
    # db.add(new_session)
    
    # [GAMIFICATION DISABLED]
    # profile = current_user.profile
    # if xp_gained > 0:
    #     profile.xp += xp_gained
    #     
    # # Streak Calculation (Only updates on first successful practice of the day)
    # if session_in.score >= 50 and profile.last_active_date != today:
    #     if profile.last_active_date is None:
    #         profile.current_streak = 1
    #         profile.longest_streak = 1
    #     else:
    #         delta = today - profile.last_active_date
    #         if delta.days == 1:
    #             profile.current_streak += 1
    #             if profile.current_streak > profile.longest_streak:
    #                 profile.longest_streak = profile.current_streak
    #         elif delta.days > 1:
    #             profile.current_streak = 1
    #             
    #     # Streak XP Bonus: current days in streak * 100
    #     streak_bonus = profile.current_streak * 100
    #     profile.xp += streak_bonus
    #     profile.last_active_date = today
    #     
    # # Calculate level based on XP (100 XP per level)
    # new_level = (profile.xp // 100) + 1
    # profile.current_level = new_level
    
    # db.commit()
    # db.refresh(new_session)
    
    return {
        "id": "mock-session-id",
        "sign_id": session_in.sign_id,
        "score": session_in.score,
        "completed_at": datetime.datetime.now(),
        "sign_name": sign.sign_name
    }

# BROWSER LOGS ROUTE
@app.post("/api/log-error")
def log_browser_error(data: dict = Body(...)):
    print(f"\n[!!! BROWSER ERROR REPORT !!!] {data}\n")
    return {"status": "logged"}

# Serving the static frontend SPA files
# The static folder will be at app/static
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
