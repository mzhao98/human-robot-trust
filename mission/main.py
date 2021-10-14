from mission import app, templates
from fastapi import Request, status
from fastapi.responses import HTMLResponse
from starlette.applications import Starlette
from starlette.routing import Route
from sse_starlette.sse import EventSourceResponse

from sqlalchemy.orm import Session
from fastapi import Depends, FastAPI, HTTPException

from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from typing import Optional
import time, json
import zmq
from fastapi import FastAPI, WebSocket
import pandas as pd
from typing import List
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from . import models, schemas, crud
from .db import ENGINE, SessionLocal


import csv
import json

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()




@app.get("/trust_game_1", response_class=HTMLResponse)
async def make_remind_1(request: Request):
    return templates.TemplateResponse("trust_game_1.html", {"request": request})




@app.post("/game_play", response_model=schemas.Game)
async def create_game(game: schemas.GameCreate, db: Session = Depends(get_db)):
    return crud.create_game(db=db, game=game)

@app.post("/record_survey", response_model=schemas.Game)
async def create_survey(survey: schemas.SurveyCreate, db: Session = Depends(get_db)):
    return crud.create_survey(db=db, survey=survey)

