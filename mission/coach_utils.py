import sys, json, numpy as np
# from py_files.refine_level_2_instructions import *
from mission.static.js.py_files.refine_level_2_instructions import *

def output_curr_room(input_x, input_y):


    player_x = int(input_x)-1
    player_y = int(input_y)-1
    print(player_x, player_y)

    Lvl2_Coach = Level_2_Instruction()
    room_to_go = Lvl2_Coach.get_room(player_x, player_y)

    return room_to_go