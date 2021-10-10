from sqlalchemy.orm import Session
from . import models, schemas
from sqlalchemy import and_, or_, func

def get_episode_by_uid(db: Session, uid: str):
    query = db.query(func.count(models.Game.userid)).filter(and_(models.Game.userid == uid, models.Game.time_spent=='start'))
    return query.scalar()


def create_game(db: Session, game: schemas.GameCreate):
    db_game = models.Game(userid=game.userid, episode=game.episode, saving_bool=game.saving_bool, \
        victim_pos=game.victim_pos, num_step=game.num_step, time_spent=game.time_spent, \
        trajectory=game.trajectory, advice_message=game.advice_message, condition=game.condition, \
                          player_score=game.player_score, quiz_score=game.quiz_score, survey_key=game.survey_key)
    db.add(db_game)
    db.commit()
    db.refresh(db_game)
    return db_game

def create_survey(db: Session, survey: schemas.SurveyCreate):
    db_survey = models.Survey(userid=survey.userid, episode=survey.episode, condition=survey.condition, \
                          player_score=survey.player_score, quiz_score=survey.quiz_score, survey_key=survey.survey_key,\
                          question=survey.question, response=survey.response)
    db.add(db_survey)
    db.commit()
    db.refresh(db_survey)
    return db_survey