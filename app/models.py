import uuid
from sqlalchemy import Column, String, Integer, ForeignKey, Date, DateTime, Text, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    sessions = relationship("PracticeSession", back_populates="user", cascade="all, delete-orphan")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    xp = Column(Integer, default=0)
    current_level = Column(Integer, default=1)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    last_active_date = Column(Date, nullable=True)
    avatar_url = Column(String(255), nullable=True)

    user = relationship("User", back_populates="profile")


class SignDictionary(Base):
    __tablename__ = "sign_dictionary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sign_name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    animation_data = Column(JSON, nullable=False)  # JSON container for bone rotation quaternions
    description = Column(Text, nullable=True)

    lesson_items = relationship("LessonItem", back_populates="sign", cascade="all, delete-orphan")
    sessions = relationship("PracticeSession", back_populates="sign")


class Lesson(Base):
    __tablename__ = "lessons"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(100), nullable=False)
    category = Column(String(50), nullable=False)
    difficulty_level = Column(String(20), nullable=False)  # 'beginner', 'intermediate', 'advanced'
    description = Column(Text, nullable=True)

    items = relationship("LessonItem", back_populates="lesson", cascade="all, delete-orphan")


class LessonItem(Base):
    __tablename__ = "lesson_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    lesson_id = Column(Integer, ForeignKey("lessons.id", ondelete="CASCADE"), nullable=False)
    sign_id = Column(Integer, ForeignKey("sign_dictionary.id", ondelete="CASCADE"), nullable=False)
    sort_order = Column(Integer, nullable=False)

    lesson = relationship("Lesson", back_populates="items")
    sign = relationship("SignDictionary", back_populates="lesson_items")


class PracticeSession(Base):
    __tablename__ = "practice_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    sign_id = Column(Integer, ForeignKey("sign_dictionary.id"), nullable=True)
    score = Column(Integer, nullable=False)  # 0-100 score
    completed_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="sessions")
    sign = relationship("SignDictionary", back_populates="sessions")
