from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username_or_email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    user_id: Optional[str] = None

class UserProfileResponse(BaseModel):
    user_id: str
    username: str
    email: str
    xp: int
    current_level: int
    current_streak: int
    longest_streak: int
    last_active_date: Optional[date] = None
    avatar_url: Optional[str] = None

    class Config:
        from_attributes = True

class SignResponse(BaseModel):
    id: int
    sign_name: str
    category: str
    description: Optional[str] = None
    animation_data: Dict[str, Any]

    class Config:
        from_attributes = True

class LessonItemResponse(BaseModel):
    id: int
    sign_id: int
    sort_order: int
    sign: SignResponse

    class Config:
        from_attributes = True

class LessonResponse(BaseModel):
    id: int
    title: str
    category: str
    difficulty_level: str
    description: Optional[str] = None
    items: List[LessonItemResponse]

    class Config:
        from_attributes = True

class PracticeSessionCreate(BaseModel):
    sign_id: int
    score: int = Field(..., ge=0, le=100)

class PracticeSessionResponse(BaseModel):
    id: str
    sign_id: Optional[int]
    score: int
    completed_at: datetime
    sign_name: Optional[str] = None

    class Config:
        from_attributes = True
