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
from .coach_utils import output_curr_room
from typing import List
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from . import models, schemas, crud
from .db import ENGINE, SessionLocal
from mission.static.js.py_files.refine_level_2_instructions import *
from mission.static.js.py_files.refine_level_2_instructions_map2 import *
#
#
# Lvl2_Coach = Level_2_Instruction()
# Lvl2_Coach_Map2 = Map_2_Instructor()

async def instantiate_coaches(level):
    if level == 1:
        coach = Level_2_Instruction()
    else:
        coach = Map_2_Instructor()
    return coach


# complexity_level = 2


import csv
import json

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# class ConnectionManager:
#     def __init__(self):
#         self.connections: List[WebSocket] = []
#
#     async def connect(self, websocket: WebSocket):
#         await websocket.accept()
#         self.connections.append(websocket)
#
#     async def broadcast(self, data: str):
#         for connection in self.connections:
#             await connection.send_text(data)
#
# manager = ConnectionManager()

@app.get("/trust_game_1", response_class=HTMLResponse)
async def make_remind_1(request: Request):
    return templates.TemplateResponse("trust_game_1.html", {"request": request})




@app.get("/ad", response_class=HTMLResponse)
async def make_ad(request: Request):
    return templates.TemplateResponse("ad.html", {"request": request})

@app.get("/end_not_participated", response_class=HTMLResponse)
async def make_end_not_participated(request: Request):
    return templates.TemplateResponse("thanks_did_not_participate.html", {"request": request})

@app.get("/end_participated", response_class=HTMLResponse)
async def make_end_partipated(request: Request):
    return templates.TemplateResponse("thanks_participated.html", {"request": request})

@app.get("/consent", response_class=HTMLResponse)
async def make_consent(request: Request):
    return templates.TemplateResponse("consent.html", {"request": request})

@app.get("/overview", response_class=HTMLResponse)
async def make_consent(request: Request):
    return templates.TemplateResponse("overview.html", {"request": request})

@app.get("/how_to_play", response_class=HTMLResponse)
async def make_consent(request: Request):
    return templates.TemplateResponse("howtoplay.html", {"request": request})

@app.get("/name", response_class=HTMLResponse)
async def make_name(request: Request):
    return templates.TemplateResponse("name.html", {"request": request})

@app.get("/instruct_1", response_class=HTMLResponse)
async def make_instruct_1(request: Request):
    return templates.TemplateResponse("instruct-1.html", {"request": request})

# @app.get("/instruct-2", response_class=HTMLResponse)
# async def make_instruct_2(request: Request):
#     return templates.TemplateResponse("instruct-2.html", {"request": request})

@app.get("/debrief", response_class=HTMLResponse)
async def make_debrief(request: Request):
    return templates.TemplateResponse("debriefing.html", {"request": request})

@app.get("/complete", response_class=HTMLResponse)
async def make_complete(request: Request):
    return templates.TemplateResponse("complete.html", {"request": request})

@app.get("/remind_1", response_class=HTMLResponse)
async def make_remind_1(request: Request):
    return templates.TemplateResponse("remind_1.html", {"request": request})

@app.get("/remind_2", response_class=HTMLResponse)
async def make_remind_1(request: Request):
    return templates.TemplateResponse("remind_2.html", {"request": request})

@app.get("/instruct_2", response_class=HTMLResponse)
async def make_instruct_2(request: Request):
    return templates.TemplateResponse("debrief.html", {"request": request})

@app.get("/post_game_1", response_class=HTMLResponse)
async def make_post_game_1(request: Request):
    return templates.TemplateResponse("post_game_1.html", {"request": request})

@app.get("/post_game_1b", response_class=HTMLResponse)
async def make_post_game_1(request: Request):
    return templates.TemplateResponse("post_game_1_emotions.html", {"request": request})

@app.get("/post_game_2", response_class=HTMLResponse)
async def make_post_game_1(request: Request):
    return templates.TemplateResponse("post_game_2.html", {"request": request})

@app.get("/post_game_2b", response_class=HTMLResponse)
async def make_post_game_1(request: Request):
    return templates.TemplateResponse("post_game_2_emotions.html", {"request": request})

@app.get("/comparison", response_class=HTMLResponse)
async def make_post_game_1(request: Request):
    return templates.TemplateResponse("comparison.html", {"request": request})

@app.get("/post_game_between_games", response_class=HTMLResponse)
async def make_post_game_1(request: Request):
    return templates.TemplateResponse("post_game_go_to_round2.html", {"request": request})


@app.get("/practice", response_class=HTMLResponse)
async def make_game(request: Request):
    return templates.TemplateResponse("minecraft_practice_round.html", {"request": request})

@app.get("/mission_1", response_class=HTMLResponse)
async def make_game(request: Request, Lvl2_Coach: Level_2_Instruction = Depends(Level_2_Instruction)):
    return templates.TemplateResponse("minecraft_game.html", {"request": request})

@app.get("/mission_2", response_class=HTMLResponse)
async def make_game(request: Request, Lvl2_Coach_Map2: Map_2_Instructor = Depends(Map_2_Instructor)):
    return templates.TemplateResponse("minecraft_game_2.html", {"request": request})

@app.get("/mission_1_round2", response_class=HTMLResponse)
async def make_game(request: Request, Lvl2_Coach: Level_2_Instruction = Depends(Level_2_Instruction)):
    return templates.TemplateResponse("minecraft_game1_round2.html", {"request": request})

@app.get("/mission_2_round2", response_class=HTMLResponse)
async def make_game(request: Request, Lvl2_Coach_Map2: Map_2_Instructor = Depends(Map_2_Instructor)):
    return templates.TemplateResponse("minecraft_game2_round2.html", {"request": request})

@app.get("/quiz_1", response_class=HTMLResponse)
async def make_quiz(request: Request):
    return templates.TemplateResponse("map_quiz_1.html", {"request": request})


@app.get("/study_1", response_class=HTMLResponse)
async def make_study(request: Request):
    return templates.TemplateResponse("map_study_1.html", {"request": request})

@app.get("/quiz_2", response_class=HTMLResponse)
async def make_quiz(request: Request):
    return templates.TemplateResponse("map_quiz_2.html", {"request": request})


@app.get("/study_2", response_class=HTMLResponse)
async def make_study(request: Request):
    return templates.TemplateResponse("map_study_2.html", {"request": request})

@app.get("/game_over", response_class=HTMLResponse)
async def make_game_over(request: Request):
    return templates.TemplateResponse("game_over.html", {"request": request})

@app.get("/opinions", response_class=HTMLResponse)
async def make_study(request: Request):
    return templates.TemplateResponse("review_study.html", {"request": request})

@app.post("/game_play", response_model=schemas.Game)
async def create_game(game: schemas.GameCreate, db: Session = Depends(get_db)):
    return crud.create_game(db=db, game=game)

@app.post("/record_survey", response_model=schemas.Game)
async def create_survey(survey: schemas.SurveyCreate, db: Session = Depends(get_db)):
    return crud.create_survey(db=db, survey=survey)


@app.websocket("/position/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, Lvl2_Coach: Level_2_Instruction = Depends(Level_2_Instruction)):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        room_name = get_room_name(Lvl2_Coach, data)
        await websocket.send_text(f"Position: {data}, Room: {room_name}")


@app.websocket("/room/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id:str, Lvl2_Coach: Level_2_Instruction = Depends(Level_2_Instruction)):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        room_name = get_room_name(Lvl2_Coach, data)
        await websocket.send_text(f"{room_name}")


@app.websocket("/mturk")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        mturk_key = get_mturk_key(data)
        await websocket.send_text(f"{mturk_key}")


@app.websocket("/advice/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id:str, Lvl2_Coach: Level_2_Instruction = Depends(Level_2_Instruction)):

    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        advice = get_advice(Lvl2_Coach, data)
        await websocket.send_text(f"{advice}")


@app.websocket("/position2/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id:str, Lvl2_Coach_Map2: Map_2_Instructor = Depends(Map_2_Instructor)):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        room_name = get_room_name_map2(Lvl2_Coach_Map2, data)
        await websocket.send_text(f"Position: {data}, Room: {room_name}")


@app.websocket("/room2/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id:str, Lvl2_Coach_Map2: Map_2_Instructor = Depends(Map_2_Instructor)):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        room_name = get_room_name_map2(Lvl2_Coach_Map2, data)
        await websocket.send_text(f"{room_name}")


@app.websocket("/advice2/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id:str, Lvl2_Coach_Map2: Map_2_Instructor = Depends(Map_2_Instructor)):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        advice = get_advice_map2(Lvl2_Coach_Map2, data)
        await websocket.send_text(f"{advice}")


def get_mturk_key(data):
    df = pd.read_csv('./mission/static/mturk/mturk_keys.csv', index_col=0)
    mturk_key = ""
    for index, row in df.iterrows():
        if row['worker_id'] == 'None':
            mturk_key = row['key']
            df.loc[index, 'worker_id'] = data
            break
    df.to_csv('./mission/static/mturk/mturk_keys.csv')
    # print("mturk_key", mturk_key)
    return mturk_key

def convert_dict_to_usable(input_dict):
    # const victim_save_add = {
    #             color: 'yellow',
    #             x: this.map.worldToTileX(player.x),
    #             y: this.map.worldToTileY(player.y),
    #           };
    input_dict_usable = {}
    for key in input_dict:
        input_dict_usable[int(key)] = {}
        input_dict_usable[int(key)]['color'] = input_dict[key]['color']
        input_dict_usable[int(key)]['x'] = int(input_dict[key]['x'])
        input_dict_usable[int(key)]['y'] = int(input_dict[key]['y'])
    return input_dict_usable

def get_room_name(Lvl2_Coach, data):
    location = json.loads(data)
    x_loc = int(location['x'])-1
    y_loc = int(location['y'])-1
    room_to_go = Lvl2_Coach.get_room(x_loc, y_loc)
    return room_to_go

def get_advice(Lvl2_Coach, data):
    input_data = json.loads(data)

    player_x = int(input_data['x'])-1
    player_y = int(input_data['y'])-1
    victim_save_record = convert_dict_to_usable(input_data['record'])
    player_heading = int(input_data['heading'])
    coach_id = int(input_data['coach'])
    past_traj = input_data['past']
    game_time = input_data['time']

    Lvl2_Coach.update_victims_record(victim_save_record)
    next_victim_to_save_id = Lvl2_Coach.victim_path_details[len(victim_save_record)]['id']
    for i in range(len(Lvl2_Coach.victim_path_details)):
        if game_time <= 120:
            if Lvl2_Coach.victim_path_details[i]['color'] == 'yellow':
                continue
        if Lvl2_Coach.victim_path_details[i]['saved_state'] == False:
            next_victim_to_save_id = Lvl2_Coach.victim_path_details[i]['id']
            break


    target_location = Lvl2_Coach.id_to_goal_tuple[next_victim_to_save_id]
    start_location = (player_x, player_y)

    if coach_id == 1:
        advice_output = Lvl2_Coach.generate_level_1_instructions(start_location, target_location,
                                                                           player_heading,
                                                                           victim_save_record)
    elif coach_id == 2:
        advice_output, final_dest = Lvl2_Coach.generate_level_2_instructions(start_location, target_location,
                                                                           player_heading,
                                                                           victim_save_record, past_traj)
    elif coach_id == 4:
        advice_output = Lvl2_Coach.generate_adaptive_instructions(start_location, target_location,
                                                                           player_heading,
                                                                           victim_save_record, past_traj, next_victim_to_save_id,
                                                                            game_time)

    else:
        advice_output = Lvl2_Coach.generate_level_3_instructions(start_location, target_location, player_heading,
                                                                       victim_save_record)

    return advice_output


def get_room_name_map2(Lvl2_Coach_Map2, data):
    location = json.loads(data)
    x_loc = int(location['x'])-1
    y_loc = int(location['y'])-1
    room_to_go = Lvl2_Coach_Map2.get_room(x_loc, y_loc)
    return room_to_go

def get_advice_map2(Lvl2_Coach_Map2, data):
    input_data = json.loads(data)
    # print("input_data", input_data)

    player_x = int(input_data['x']) - 1
    player_y = int(input_data['y']) - 1
    victim_save_record = convert_dict_to_usable(input_data['record'])
    player_heading = int(input_data['heading'])
    coach_id = int(input_data['coach'])
    past_traj = input_data['past']
    game_time = input_data['time']
    # print("past_traj", past_traj)
    level = "2"

    # print("HEADING: ", player_heading)
    Lvl2_Coach_Map2.update_victims_record(victim_save_record)
    next_victim_to_save_id = Lvl2_Coach_Map2.victim_path_details[len(victim_save_record)]['id']
    for i in range(len(Lvl2_Coach_Map2.victim_path_details)):
        if game_time <= 120:
            if Lvl2_Coach_Map2.victim_path_details[i]['color'] == 'yellow':
                continue
        if Lvl2_Coach_Map2.victim_path_details[i]['saved_state'] == False:
            next_victim_to_save_id = Lvl2_Coach_Map2.victim_path_details[i]['id']
            break

    # print("TARGET VICTIM = ", Lvl2_Coach.victim_path_details[next_victim_to_save_id])
    target_location = Lvl2_Coach_Map2.id_to_goal_tuple[next_victim_to_save_id]
    start_location = (player_x, player_y)


    if coach_id == 1:
        advice_output = Lvl2_Coach_Map2.generate_level_1_instructions(start_location, target_location,
                                                                           player_heading,
                                                                           victim_save_record,to_plot = False)
    elif coach_id == 2:
        advice_output, final_dest = Lvl2_Coach_Map2.generate_level_2_instructions(start_location, target_location,
                                                                           player_heading,
                                                                           victim_save_record, past_traj)
    elif coach_id == 4:
        advice_output = Lvl2_Coach_Map2.generate_adaptive_instructions(start_location, target_location,
                                                                           player_heading,
                                                                           victim_save_record, past_traj, next_victim_to_save_id,
                                                                       game_time)

    else:
        advice_output = Lvl2_Coach_Map2.generate_level_3_instructions(start_location, target_location, player_heading,
                                                                       victim_save_record)
    # print(advice_output)
    # if player_x % 2 == 0:
    #     advice_output = "even"
    # else:
    #     advice_output = "odd"

    return advice_output
