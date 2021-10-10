from typing import List
from pydantic import BaseModel


class GameBase(BaseModel):
    userid: str

class GameCreate(GameBase):
    episode: int
    saving_bool: str
    victim_pos: str
    num_step: int
    time_spent: str
    trajectory: str
    advice_message: str
    condition: int
    player_score: int
    quiz_score: int
    survey_key: str

class Game(GameBase):
    id: int
    userid: str

    class Config:
        orm_mode = True


class SurveyBase(BaseModel):
    userid: str

class SurveyCreate(SurveyBase):
    episode: int
    condition: int
    player_score: int
    quiz_score: int
    survey_key: str
    question: str
    response: str

class Survey(SurveyBase):
    id: int
    userid: str

    class Config:
        orm_mode = True




