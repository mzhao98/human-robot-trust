from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .db import Base

class Game(Base):
    __tablename__ = "game"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    userid = Column(String, nullable=False)
    episode = Column(Integer, nullable=False)
    saving_bool = Column(String)
    victim_pos = Column(String)
    num_step = Column(Integer)
    time_spent = Column(String)
    trajectory = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    advice_message = Column(String)
    condition = Column(Integer, nullable=False)
    player_score = Column(Integer)
    quiz_score = Column(Integer)
    survey_key = Column(String)


class Survey(Base):
    __tablename__ = "survey"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    userid = Column(String, nullable=False)
    episode = Column(Integer, nullable=False)
    condition = Column(Integer, nullable=False)
    player_score = Column(Integer)
    quiz_score = Column(Integer)
    survey_key = Column(String)
    question = Column(String)
    response = Column(String)



