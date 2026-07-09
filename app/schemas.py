from pydantic import BaseModel
from typing import List, Optional


class SignResponse(BaseModel):
    id: int
    sign_name: str
    category: str
    description: Optional[str] = None

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
