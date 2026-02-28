from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, JSON, Float
from sqlalchemy.orm import relationship, DeclarativeBase
from datetime import datetime

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String) # panelist, admin
    organization_id = Column(String)

class Candidate(Base):
    __tablename__ = "candidates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    status = Column(String, default="Pending") # Pending, Interviewing, Evaluated
    applied_role = Column(String)
    
    assessments = relationship("Assessment", back_populates="candidate")

class QuestionBank(Base):
    __tablename__ = "question_bank"
    id = Column(Integer, primary_key=True, index=True)
    category = Column(String) # Skill, Training, Attitude, Results
    question_text = Column(String)
    options = Column(JSON)
    correct_answer = Column(String)
    source = Column(String) # standard, kaggle

class Assessment(Base):
    __tablename__ = "assessments"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    user_id = Column(Integer, ForeignKey("users.id")) # Interviewer
    access_key = Column(String, unique=True, index=True) # For candidate access
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    state = Column(JSON) # Current progress state
    
    candidate = relationship("Candidate", back_populates="assessments")
    responses = relationship("Response", back_populates="assessment")

class Response(Base):
    __tablename__ = "responses"
    id = Column(Integer, primary_key=True, index=True)
    assessment_id = Column(Integer, ForeignKey("assessments.id"))
    question_text = Column(String)
    transcript = Column(String) # Candidate's written/recorded answer
    score = Column(Integer)
    ai_generated = Column(Integer, default=0) # Boolean 0/1 or actual bool
    ai_feedback = Column(String)
    
    assessment = relationship("Assessment", back_populates="responses")
