from sqlalchemy import Column, String, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base


class SignDictionary(Base):
    __tablename__ = "sign_dictionary"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sign_name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=True)

    lesson_items = relationship("LessonItem", back_populates="sign", cascade="all, delete-orphan")


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
