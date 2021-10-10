import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import heapq
import pickle as pkl
# import pygame, sys, time, random

import sys, time, random
sys.path.append('./mission/js/py_files')


from mission.static.js.py_files.a_star import Search, Decision_Point_Search
from mission.static.js.py_files.gameboard_utils import *
from mission.static.js.py_files.settings import *



def decision_point_dict():
    decision_points = {
        -1: {
            'type': 'start',
            'location': (6, 5),
            'paired_doors': [],
        },
        0: {
            'type': 'door',
            'location': (25, 8),
            'paired_doors': [],
        },
        1: {
            'type': 'door',
            'location': (25, 13),
            'paired_doors': [],
        },
        2: {
            'type': 'door',
            'location': (31, 7),
            'extended_locs': [(29, 7), (30, 7), (31, 7), (32, 7), (33, 7)],
            'paired_doors': [],
        },
        3: {
            'type': 'door',
            'location': (37, 7),
            'paired_doors': [],
        },
        4: {
            'type': 'door',
            'location': (54, 7),
            'paired_doors': [],
        },
        5: {
            'type': 'door',
            'location': (70, 7),
            'paired_doors': [],
        },
        6: {
            'type': 'door',
            'location': (84, 8),
            'paired_doors': [],
        },
        7: {
            'type': 'entrance',
            'location': (25, 10),
            'paired_doors': [0],
        },
        7.5: {
            'type': 'entrance',
            'location': (25, 12),
            'paired_doors': [1],
        },
        8: {
            'type': 'entrance',
            'location': (31, 10),
            'extended_locs': [(29, 10), (30, 10), (31, 10), (32, 10), (33, 10)],
            'paired_doors': [2],
        },
        9: {
            'type': 'entrance',
            'location': (37, 10),
            'paired_doors': [3],
        },
        10: {
            'type': 'entrance',
            'location': (54, 10),
            'paired_doors': [4],
        },
        11: {
            'type': 'entrance',
            'location': (70, 10),
            'paired_doors': [5],
        },
        12: {
            'type': 'entrance',
            'location': (82, 8),
            'paired_doors': [6],
        },
        13: {
            'type': 'door',
            'location': (41, 12),
            'paired_doors': [],
        },
        14: {
            'type': 'entrance',
            'location': (41, 10),
            'paired_doors': [13],
        },
        15: {
            'type': 'door',
            'location': (70, 15),
            'extended_locations': [(70, 15), (70, 14)],
            'paired_doors': [],
        },
        16: {
            'type': 'entrance',
            'location': (72, 15),
            'extended_locations': [(72, 15), (72, 14)],
            'paired_doors': [15],
        },
        17: {
            'type': 'door',
            'location': (39, 18),
            'paired_doors': [],
        },
        18: {
            'type': 'entrance',
            'location': (37, 18),
            'paired_doors': [17],
        },
        19: {
            'type': 'door',
            'location': (43, 20),
            'paired_doors': [],
        },
        20: {
            'type': 'entrance',
            'location': (45, 20),
            'paired_doors': [19],
        },
        21: {
            'type': 'door',
            'location': (43, 24),
            'paired_doors': [],
        },
        22: {
            'type': 'entrance',
            'location': (45, 24),
            'paired_doors': [21],
        },
        23: {
            'type': 'door',
            'location': (56, 22),
            'extended_locations': [(56, 22), (56, 21)],
            'paired_doors': [],
        },
        24: {
            'type': 'entrance',
            'location': (58, 22),
            'extended_locations': [(58, 22), (58, 21)],
            'paired_doors': [23],
        },
        25: {
            'type': 'door',
            'location': (56, 25),
            'extended_locations': [(56, 24), (56, 25)],
            'paired_doors': [23],
        },
        26: {
            'type': 'entrance',
            'location': (59, 25),
            'extended_locations': [(58, 24), (58, 25)],
            'paired_doors': [23],
        },
        27: {
            'type': 'door',
            'location': (75, 25),
            'paired_doors': [],
        },
        28: {
            'type': 'entrance',
            'location': (73, 25),
            'paired_doors': [27],
        },
        29: {
            'type': 'door',
            'location': (39, 27),
            'paired_doors': [],
        },
        30: {
            'type': 'entrance',
            'location': (37, 27),
            'paired_doors': [29],
        },
        31: {
            'type': 'door',
            'location': (43, 29),
            'paired_doors': [],
        },
        32: {
            'type': 'entrance',
            'location': (45, 29),
            'paired_doors': [31],
        },
        33: {
            'type': 'door',
            'location': (42, 33),
            'paired_doors': [],
        },
        34: {
            'type': 'entrance',
            'location': (44, 33),
            'paired_doors': [33],
        },
        35: {
            'type': 'door',
            'location': (61, 32),
            'extended_locations': [(61, 32), (61, 33)],
            'paired_doors': [],
        },
        36: {
            'type': 'entrance',
            'location': (58, 32),
            'extended_locations': [(58, 32), (58, 33)],
            'paired_doors': [35],
        },
        37: {
            'type': 'door',
            'location': (25, 35),
            'paired_doors': [],
        },
        38: {
            'type': 'entrance',
            'location': (25, 38),
            'paired_doors': [37],
        },
        39: {
            'type': 'door',
            'location': (76, 35),
            'paired_doors': [],
        },
        40: {
            'type': 'entrance',
            'location': (76, 38),
            'paired_doors': [39],
        },
        41: {
            'type': 'door',
            'location': (4, 40),
            'paired_doors': [],
        },
        42: {
            'type': 'entrance',
            'location': (4, 38),
            'paired_doors': [41],
        },
        43: {
            'type': 'door',
            'location': (13, 40),
            'paired_doors': [],
        },
        44: {
            'type': 'entrance',
            'location': (13, 38),
            'paired_doors': [44],
        },
        45: {
            'type': 'door',
            'location': (22, 40),
            'paired_doors': [],
        },
        46: {
            'type': 'entrance',
            'location': (22, 38),
            'paired_doors': [45],
        },
        47: {
            'type': 'door',
            'location': (31, 40),
            'paired_doors': [],
        },
        48: {
            'type': 'entrance',
            'location': (31, 38),
            'paired_doors': [47],
        },
        49: {
            'type': 'door',
            'location': (40, 40),
            'paired_doors': [],
        },
        50: {
            'type': 'entrance',
            'location': (40, 38),
            'paired_doors': [49],
        },
        51: {
            'type': 'door',
            'location': (49, 40),
            'paired_doors': [],
        },
        52: {
            'type': 'entrance',
            'location': (49, 38),
            'paired_doors': [51],
        },
        53: {
            'type': 'door',
            'location': (58, 40),
            'paired_doors': [],
        },
        54: {
            'type': 'entrance',
            'location': (58, 38),
            'paired_doors': [53],
        },
        55: {
            'type': 'door',
            'location': (67, 40),
            'paired_doors': [],
        },
        56: {
            'type': 'entrance',
            'location': (67, 38),
            'paired_doors': [55],
        },
        57: {
            'type': 'door',
            'location': (76, 40),
            'paired_doors': [],
        },
        58: {
            'type': 'entrance',
            'location': (76, 38),
            'paired_doors': [57],
        },
        59: {
            'type': 'intersection',
            'location': (19, 5),
            'paired_doors': [],
        },
        60: {
            'type': 'intersection',
            'location': (19, 10),
            'paired_doors': [],
        },
        61: {
            'type': 'intersection',
            'location': (36, 10),
            'paired_doors': [],
        },
        62: {
            'type': 'intersection',
            'location': (59, 10),
            'paired_doors': [],
        },
        63: {
            'type': 'intersection',
            'location': (72, 10),
            'paired_doors': [],
        },
        64: {
            'type': 'intersection',
            'location': (4, 38),
            'paired_doors': [],
        },
        65: {
            'type': 'intersection',
            'location': (36, 38),
            'paired_doors': [],
        },
        66: {
            'type': 'intersection',
            'location': (59, 38),
            'paired_doors': [],
        },
        67: {
            'type': 'intersection',
            'location': (72, 38),
            'paired_doors': [],
        },
        68: {
            'type': 'intersection',
            'location': (82, 38),
            'paired_doors': [],
        },

        69: {
            'type': 'victim',
            'location': (30,4),
            'victim_index': 0,
            'color': 'g',
        },
        70: {
            'type': 'victim',
            'location': (35,7),
            'victim_index': 1,
            'color': 'y',
        },
        71: {
            'type': 'victim',
            'location': (50,5),
            'victim_index': 2,
            'color': 'g',
        },
        72: {
            'type': 'victim',
            'location': (48,6),
            'victim_index': 3,
            'color': 'g',
        },
        73: {
            'type': 'victim',
            'location': (74,6),
            'victim_index': 4,
            'color': 'g',
        },
        74: {
            'type': 'victim',
            'location': (87, 4),
            'victim_index': 5,
            'color': 'g',
        },
        75: {
            'type': 'victim',
            'location': (78,21),
            'victim_index': 6,
            'color': 'y',
        },
        76: {
            'type': 'victim',
            'location': (82,33),
            'victim_index': 7,
            'color': 'g',
        },
        77: {
            'type': 'victim',
            'location': (81,48),
            'victim_index': 8,
            'color': 'g',
        },
        78: {
            'type': 'victim',
            'location': (63,47),
            'victim_index': 9,
            'color': 'g',
        },
        79: {
            'type': 'victim',
            'location': (45, 47),
            'victim_index': 10,
            'color': 'y',
        },
        80: {
            'type': 'victim',
            'location': (37, 47),
            'victim_index': 11,
            'color': 'g',
        },
        81: {
            'type': 'victim',
            'location': (27, 43),
            'victim_index': 12,
            'color': 'g',
        },
        82: {
            'type': 'victim',
            'location': (19, 47),
            'victim_index': 13,
            'color': 'y',
        },
        83: {
            'type': 'victim',
            'location': (22, 25),
            'victim_index': 14,
            'color': 'g',
        },
        84: {
            'type': 'victim',
            'location': (31, 31),
            'victim_index': 15,
            'color': 'y',
        },
        85: {
            'type': 'victim',
            'location': (40, 33),
            'victim_index': 16,
            'color': 'g',
        },
        86: {
            'type': 'victim',
            'location': (40, 20),
            'victim_index': 17,
            'color': 'g',
        },
        87: {
            'type': 'victim',
            'location': (48, 23),
            'victim_index': 18,
            'color': 'y',
        },
        88: {
            'type': 'victim',
            'location': (48, 34),
            'victim_index': 19,
            'color': 'g',
        },
        89: {
            'type': 'victim',
            'location': (67, 18),
            'victim_index': 20,
            'color': 'y',
        },
        90: {
            'type': 'victim',
            'location': (65, 30),
            'victim_index': 21,
            'color': 'y',
        },

    }
    intersections = {}
    entrances = {}
    doors = {}
    for key in decision_points:
        if decision_points[key]['type']=='intersection':
            intersections[key] = decision_points[key]
        if decision_points[key]['type']=='entrance':
            entrances[key] = decision_points[key]
        if decision_points[key]['type']=='door' or decision_points[key]['type']=='open door':
            doors[key] = decision_points[key]
    return decision_points, intersections, entrances, doors


def create_decision_points_neighbors():
    decision_neighbors = {}
    decision_neighbors[-1] = [59]
    decision_neighbors[59] = [60]
    decision_neighbors[6] = [12, 74]
    decision_neighbors[60] = [59, 7, 7.5]
    decision_neighbors[7] = [60, 0, 8, 7.5]
    decision_neighbors[7.5] = [60, 1, 8, 7]
    decision_neighbors[0] = [7]
    decision_neighbors[1] = [7.5, 37, 83, 84]
    decision_neighbors[8] = [2, 7, 7.5, 61]
    decision_neighbors[2] = [8, 69, 70]
    decision_neighbors[61] = [8,9,14,18]
    decision_neighbors[9] = [61,3,14]
    decision_neighbors[3] = [9, 71, 72]
    decision_neighbors[14] = [9,61,10,13]
    decision_neighbors[13] = [14]
    decision_neighbors[10] = [14,4,62]
    decision_neighbors[4] = [10]
    decision_neighbors[62] = [10,11,24]
    decision_neighbors[11] = [62,5,63]
    decision_neighbors[5] = [11,12, 73]
    decision_neighbors[12] = [5,6]
    decision_neighbors[63] = [11,16]
    decision_neighbors[16] = [15,28]
    decision_neighbors[15] = [16,35, 89, 90]
    decision_neighbors[28] = [16,27,67]
    decision_neighbors[27] = [28, 75]
    decision_neighbors[67] = [40,58,56,28]
    decision_neighbors[40] = [58,39, 57,68,67]
    decision_neighbors[58] = [40,39,57,67,68]
    decision_neighbors[39] = [40,58, 76]
    decision_neighbors[57] = [40,58, 77]
    decision_neighbors[68] = [40,58]
    decision_neighbors[56] = [66,67,55]
    decision_neighbors[55] = [56]
    decision_neighbors[66] = [54,56,36]
    decision_neighbors[54] = [66,53,52]
    decision_neighbors[53] = [54, 78]
    decision_neighbors[36] = [54,66,35,26]
    decision_neighbors[35] = [36,15, 90, 89]
    decision_neighbors[26] = [25,36,24]
    decision_neighbors[25] = [26, 88]
    decision_neighbors[24] = [26,62,23]
    decision_neighbors[23] = [24, 87]
    decision_neighbors[52] = [51,54,50]
    decision_neighbors[51] = [52]
    decision_neighbors[50] = [52,49,65]
    decision_neighbors[49] = [50, 79]
    decision_neighbors[48] = [65,47,38]
    decision_neighbors[65] = [50, 48, 30]
    decision_neighbors[47] = [48, 80]
    decision_neighbors[30] = [65,29,18]
    decision_neighbors[29] = [30,32]
    decision_neighbors[32] = [29,34,31]
    decision_neighbors[31] = [32]
    decision_neighbors[34] = [33,32]
    decision_neighbors[33] = [34, 86]
    decision_neighbors[18] = [30,61,17]
    decision_neighbors[17] = [18,20]
    decision_neighbors[20] = [17,19,22]
    decision_neighbors[19] = [20, 85]
    decision_neighbors[22] = [20,21]
    decision_neighbors[21] = [22]
    decision_neighbors[38] = [37,48,46]
    decision_neighbors[37] = [38,1, 83, 84]
    decision_neighbors[46] = [38,45,44]
    decision_neighbors[45] = [46, 81]
    decision_neighbors[44] = [46,64,42,43]
    decision_neighbors[43] = [44, 82]
    decision_neighbors[64] = [44,42]
    decision_neighbors[42] = [41,44,64]
    decision_neighbors[41] = [42]

    # victim connections
    decision_neighbors[69] = [2, 70]
    decision_neighbors[70] = [2,69]
    decision_neighbors[71] = [3, 72]
    decision_neighbors[72] = [71,3]
    decision_neighbors[73] = [5, 12]
    decision_neighbors[74] = [6]
    decision_neighbors[75] = [27]
    decision_neighbors[76] = [39]
    decision_neighbors[77] = [57]
    decision_neighbors[78] = [53]
    decision_neighbors[79] = [49]
    decision_neighbors[79] = [49]
    decision_neighbors[80] = [47]
    decision_neighbors[81] = [45]
    decision_neighbors[82] = [43]
    decision_neighbors[83] = [37, 84, 1]
    decision_neighbors[84] = [37, 83, 1]
    decision_neighbors[85] = [19]
    decision_neighbors[86] = [33]
    decision_neighbors[87] = [23]
    decision_neighbors[88] = [25]
    decision_neighbors[89] = [35, 90, 15]
    decision_neighbors[90] = [35, 89, 15]


    return decision_neighbors

def euclidean(tup1, tup2):
    return np.sqrt((tup1[0]-tup2[0])**2 + (tup1[1]-tup2[1])**2)


def recompute_path(current_location, target_location, id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs):
    single_search = Search(gameboard, current_location, target_location, obstacles, yellow_locs, green_locs)
    travel_path = single_search.a_star_new(current_location, target_location)
    return travel_path


def convert_path_to_level_1_orig(prev_loc, travel_path, gameboard, obstacles, yellow_locs, green_locs, doors, stairs,
                            goal_tuple_to_id, id_to_goal_tuple, reversed_complete_room_dict):
    loc_tuples = travel_path
    x_waypoints = [loc[0] for loc in loc_tuples]
    z_waypoints = [loc[1] for loc in loc_tuples]
    if x_waypoints[0] < prev_loc[0]:
        current_direction = 'W'
    elif x_waypoints[0] > prev_loc[0]:
        current_direction = 'E'
    elif z_waypoints[0] < prev_loc[1]:
        current_direction = 'N'
    elif z_waypoints[0] > prev_loc[1]:
        current_direction = 'S'
    else:
        current_direction = 'N'

    current_advice = ('Walk forward ', 0)
    advice = ''

    prev_loc = (x_waypoints[0], z_waypoints[0])
    for j in range(1, len(x_waypoints)):
        if x_waypoints[j] < prev_loc[0]:
            next_direction = 'W'
        elif x_waypoints[j] > prev_loc[0]:
            next_direction = 'E'
        elif z_waypoints[j] < prev_loc[1]:
            next_direction = 'N'
        elif z_waypoints[j] > prev_loc[1]:
            next_direction = 'S'
        else:
            next_direction = current_direction

        if next_direction == current_direction:
            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))
        else:
            if current_advice[1] > 0:
                advice += current_advice[0] + str(current_advice[1]) + ' steps. '
            if current_direction == 'E' and next_direction == 'N':
                advice += 'Turn Left. '
            if current_direction == 'W' and next_direction == 'N':
                advice += 'Turn Right. '
            if current_direction == 'S' and next_direction == 'N':
                advice += 'Turn Around. '

            if current_direction == 'E' and next_direction == 'S':
                advice += 'Turn Right. '
            if current_direction == 'W' and next_direction == 'S':
                advice += 'Turn Left. '
            if current_direction == 'N' and next_direction == 'S':
                advice += 'Turn Around. '

            if current_direction == 'N' and next_direction == 'E':
                advice += 'Turn Right. '
            if current_direction == 'W' and next_direction == 'E':
                advice += 'Turn Around. '
            if current_direction == 'S' and next_direction == 'E':
                advice += 'Turn Left. '

            if current_direction == 'N' and next_direction == 'W':
                advice += 'Turn Left. '
            if current_direction == 'E' and next_direction == 'W':
                advice += 'Turn Around. '
            if current_direction == 'S' and next_direction == 'W':
                advice += 'Turn Right. '

            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))

            current_direction = next_direction
        prev_loc = (x_waypoints[j], z_waypoints[j])
    if current_advice[1] > 0:
        advice += current_advice[0] + str(current_advice[1]) + ' steps. '
    if (x_waypoints[-1], z_waypoints[-1]) in yellow_locs or (x_waypoints[-1], z_waypoints[-1]) in green_locs:
        advice += 'Swing Medical Ax 10 Times.'
    final_advice = ' '.join([x+'. ' for x in advice.split('. ')[:5]])
    return final_advice

def convert_path_to_level_1(prev_loc, travel_path, gameboard, obstacles, yellow_locs, green_locs, doors, stairs,
                            goal_tuple_to_id, id_to_goal_tuple, reversed_complete_room_dict):
    loc_tuples = travel_path
    x_waypoints = [loc[0] for loc in loc_tuples]
    z_waypoints = [loc[1] for loc in loc_tuples]

    current_advice = ('Walk forward ', 0)
    advice = ''
    advice_list = []
    need_to_reach_destinations = []


    for j in range(1, len(x_waypoints)-1):
        prev_loc = (x_waypoints[j-1], z_waypoints[j-1])
        curr_loc = (x_waypoints[j], z_waypoints[j])
        next_loc = (x_waypoints[j + 1], z_waypoints[j + 1])

        x_diff = curr_loc[0] - prev_loc[0]
        z_diff = curr_loc[1] - prev_loc[1]
        if abs(x_diff) > abs(z_diff):
            if x_diff < 0:
                curr_direction = 'W'
            else:
                curr_direction = 'E'
        else:
            if z_diff < 0:
                curr_direction = 'N'
            else:
                curr_direction = 'S'

        x_diff = next_loc[0] - curr_loc[0]
        z_diff = next_loc[1] - curr_loc[1]
        if abs(x_diff) > abs(z_diff):
            if x_diff < 0:
                next_direction = 'W'
            else:
                next_direction = 'E'
        else:
            if z_diff < 0:
                next_direction = 'N'
            else:
                next_direction = 'S'


        if next_direction == curr_direction:
            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))
        else:
            if current_advice[1] > 0:
                advice += "Walk " + str(current_advice[1]) + ' steps '+curr_direction+'. '
                advice += "Face " + next_direction + '. '
                advice_list.append("Walk " + str(current_advice[1]) + ' steps '+curr_direction+'. ')
                advice_list.append("Face " + next_direction + '. ')
                need_to_reach_destinations.append((x_waypoints[j], z_waypoints[j]))
                need_to_reach_destinations.append((x_waypoints[j], z_waypoints[j]))

            current_advice = ('Walk forward ', 0)
            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))

    if current_advice[1] > 0:
        advice += "Walk " + str(current_advice[1]) + ' steps ' + curr_direction + '. '
        advice_list.append("Walk " + str(current_advice[1]) + ' steps ' + curr_direction + '. ')
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))
    if (x_waypoints[-1], z_waypoints[-1]) in yellow_locs or (x_waypoints[-1], z_waypoints[-1]) in green_locs:
        advice += 'Swing Medical Ax 10 Times. '
        advice_list.append('Swing Medical Ax 10 Times. ')
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))

    # advice_list = [x+'. ' for x in advice.split('. ')]

    if len(advice_list) > 3:
        final_destination = need_to_reach_destinations[2]
        advice_list = advice_list[:3]
    else:
        final_destination = (x_waypoints[-1], z_waypoints[-1])
    final_advice = ' '.join(advice_list)
    return final_advice, final_destination

def convert_path_to_level_2(prev_loc, travel_path, gameboard, obstacles, yellow_locs, green_locs, doors,
                            stairs, goal_tuple_to_id, id_to_goal_tuple, reversed_complete_room_dict):
    loc_tuples = travel_path
    x_waypoints = [loc[0] for loc in loc_tuples]
    z_waypoints = [loc[1] for loc in loc_tuples]
    if x_waypoints[0] < prev_loc[0]:
        current_direction = 'W'
    elif x_waypoints[0] > prev_loc[0]:
        current_direction = 'E'
    elif z_waypoints[0] < prev_loc[1]:
        current_direction = 'N'
    elif z_waypoints[0] > prev_loc[1]:
        current_direction = 'S'
    else:
        current_direction = 'N'

    current_advice = ('Walk forward ', 0)
    advice = ''

    prev_loc = (x_waypoints[0], z_waypoints[0])


    for j in range(1, len(x_waypoints)):
        if x_waypoints[j] < prev_loc[0]:
            next_direction = 'W'
        elif x_waypoints[j] > prev_loc[0]:
            next_direction = 'E'
        elif z_waypoints[j] < prev_loc[1]:
            next_direction = 'N'
        elif z_waypoints[j] > prev_loc[1]:
            next_direction = 'S'
        else:
            next_direction = current_direction

        if next_direction == current_direction:
            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))
        else:
            if current_advice[1] > 0:
                nearest = min(list(reversed_complete_room_dict.keys()),
                              key=lambda c: (c[0] - z_waypoints[j]) ** 2 + (c[1] - x_waypoints[j]) ** 2)
                next_room = reversed_complete_room_dict[nearest]
                num_room_changes = 0
                prev_room = None
                if current_direction == 'N':
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]+10) ** 2 + (c[1] - z_waypoints[j]+(current_advice[1]-1-r)) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j], z_waypoints[j]+(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes += 1
                                prev_room = current_room
                    if 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes == 0:
                            advice += 'Proceed to the door immediately '
                        else:
                            advice += 'Proceed to ' + str(num_room_changes) + 'th door '
                    else:
                        advice += 'Proceed to ' + 'hallway '
                    if next_direction == 'E':
                        advice += 'on your right. '
                    if next_direction == 'W':
                        advice += 'on your left. '
                    if next_direction == 'S':
                        advice += 'behind you. '
                if current_direction == 'S':
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]+10) ** 2 + (
                                                  c[1] - z_waypoints[j] - (current_advice[1] - 1 - r)) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j], z_waypoints[j]-(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes += 1
                                prev_room = current_room
                    if 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes == 0:
                            advice += 'Proceed to the door immediately '
                        else:
                            advice += 'Proceed to ' + str(num_room_changes) + 'th door '
                    else:
                        advice += 'Proceed to ' + 'hallway '
                    if next_direction == 'E':
                        advice += 'on your left. '
                    if next_direction == 'W':
                        advice += 'on your right. '
                    if next_direction == 'N':
                        advice += 'behind you. '
                if current_direction == 'W':
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]+(current_advice[1]-1-r)) ** 2 + (
                                                  c[1] - z_waypoints[j]+10) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j]+(current_advice[1]-1-r), z_waypoints[j])]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes += 1
                                prev_room = current_room
                    if 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes == 0:
                            advice += 'Proceed to the door immediately '
                        else:
                            advice += 'Proceed to ' + str(num_room_changes) + 'th door '
                    else:
                        advice += 'Proceed to ' + 'hallway '
                    if next_direction == 'N':
                        advice += 'on your right. '
                    if next_direction == 'S':
                        advice += 'on your left. '
                    if next_direction == 'W':
                        advice += 'behind you. '

                if current_direction == 'E':
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j] - (current_advice[1] - 1 - r)) ** 2 + (
                                              c[1] - z_waypoints[j]+10) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j]-(current_advice[1]-1-r), z_waypoints[j]-(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes += 1
                                prev_room = current_room
                    if 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes == 0:
                            advice += 'Proceed to the door immediately '
                        else:
                            advice += 'Proceed to ' + str(num_room_changes) + 'th door '
                    else:
                        advice += 'Proceed to ' + 'hallway '
                    if next_direction == 'N':
                        advice += 'on your left. '
                    if next_direction == 'S':
                        advice += 'on your right. '
                    if next_direction == 'W':
                        advice += 'behind you. '

            advice += 'Orient ' + str(next_direction) + '. '

            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))

            current_direction = next_direction
        prev_loc = (x_waypoints[j], z_waypoints[j])
    # if current_advice[1] > 0:
    #     advice += current_advice[0] + str(current_advice[1]) + ' steps. '
    if (x_waypoints[-1], z_waypoints[-1]) in yellow_locs or (x_waypoints[-1], z_waypoints[-1]) in green_locs:
        advice += 'Triage Victim.'
    final_advice = ' '.join([x+'. ' for x in advice.split('. ')[:2]])
    return final_advice

def convert_path_to_level_2_update(prev_loc, travel_path, gameboard, obstacles, yellow_locs, green_locs, doors,
                            stairs, goal_tuple_to_id, id_to_goal_tuple, reversed_complete_room_dict):
    loc_tuples = travel_path
    x_waypoints = [loc[0] for loc in loc_tuples]
    z_waypoints = [loc[1] for loc in loc_tuples]
    if x_waypoints[0] < prev_loc[0]:
        current_direction = 'W'
    elif x_waypoints[0] > prev_loc[0]:
        current_direction = 'E'
    elif z_waypoints[0] < prev_loc[1]:
        current_direction = 'N'
    elif z_waypoints[0] > prev_loc[1]:
        current_direction = 'S'
    else:
        current_direction = 'N'

    current_advice = ('Walk forward ', 0)
    advice = ''

    prev_loc = (x_waypoints[0], z_waypoints[0])
    for j in range(1, len(x_waypoints)):
        if x_waypoints[j] < prev_loc[0]:
            next_direction = 'W'
        elif x_waypoints[j] > prev_loc[0]:
            next_direction = 'E'
        elif z_waypoints[j] < prev_loc[1]:
            next_direction = 'N'
        elif z_waypoints[j] > prev_loc[1]:
            next_direction = 'S'
        else:
            next_direction = current_direction

        if next_direction == current_direction:
            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))
        else:
            if current_advice[1] > 0:
                nearest = min(list(reversed_complete_room_dict.keys()),
                              key=lambda c: (c[0] - z_waypoints[j]) ** 2 + (c[1] - x_waypoints[j]) ** 2)
                next_room = reversed_complete_room_dict[nearest]


                if current_direction == 'N':
                    num_room_changes_left = 0
                    num_room_changes_right = 0
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]+10) ** 2 + (c[1] - z_waypoints[j]+(current_advice[1]-1-r)) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j], z_waypoints[j]+(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_right += 1
                                prev_room = current_room
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]-10) ** 2 + (c[1] - z_waypoints[j]+(current_advice[1]-1-r)) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j], z_waypoints[j]+(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_left += 1
                                prev_room = current_room

                    if 'hallway' in current_room:
                        advice += 'Proceed to ' + 'hallway '
                    elif 'bridge' in current_room:
                        advice += 'Proceed to ' + 'bridge '
                    elif 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes_left == 0 and num_room_changes_right == 0:
                            advice += 'Proceed to the door immediately '
                    else:
                        if next_direction == 'E':
                            advice += 'Proceed to '
                            if num_room_changes_right == 1:
                                advice += 'first'
                            if num_room_changes_right == 2:
                                advice += 'second'
                            if num_room_changes_right == 3:
                                advice += 'third'
                            advice += ' door on your right. '
                        if next_direction == 'W':
                            advice += 'Proceed to '
                            if num_room_changes_left == 1:
                                advice += 'first'
                            if num_room_changes_left == 2:
                                advice += 'second'
                            if num_room_changes_left == 3:
                                advice += 'third'
                            advice += ' door on your left. '
                        if next_direction == 'S':
                            advice += 'Turn around. '
                            # advice += 'behind you. '
                if current_direction == 'S':
                    num_room_changes_left = 0
                    num_room_changes_right = 0
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]+10) ** 2 + (
                                                  c[1] - z_waypoints[j] - (current_advice[1] - 1 - r)) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j], z_waypoints[j]-(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_right += 1
                                prev_room = current_room
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]-10) ** 2 + (
                                                  c[1] - z_waypoints[j] - (current_advice[1] - 1 - r)) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j], z_waypoints[j]-(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_left += 1
                                prev_room = current_room
                    if 'hallway' in current_room:
                        advice += 'Proceed to ' + 'hallway '
                    elif 'bridge' in current_room:
                        advice += 'Proceed to ' + 'bridge '
                    elif 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes_left == 0 and num_room_changes_right == 0:
                            advice += 'Proceed to the door immediately '
                    else:
                        if next_direction == 'W':
                            advice += 'Proceed to '
                            if num_room_changes_right == 1:
                                advice += 'first'
                            if num_room_changes_right == 2:
                                advice += 'second'
                            if num_room_changes_right == 3:
                                advice += 'third'
                            advice += ' door on your right. '
                        if next_direction == 'E':
                            advice += 'Proceed to '
                            if num_room_changes_left == 1:
                                advice += 'first'
                            if num_room_changes_left == 2:
                                advice += 'second'
                            if num_room_changes_left == 3:
                                advice += 'third'
                            advice += ' door on your left. '
                        if next_direction == 'N':
                            advice += 'Turn around. '
                            # advice += 'behind you. '
                if current_direction == 'W':
                    num_room_changes_up = 0
                    num_room_changes_down = 0
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j]+(current_advice[1]-1-r)) ** 2 + (
                                                  c[1] - z_waypoints[j]+10) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j]+(current_advice[1]-1-r), z_waypoints[j])]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_down += 1
                                prev_room = current_room
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j] + (current_advice[1] - 1 - r)) ** 2 + (
                                              c[1] - z_waypoints[j] - 10) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j]+(current_advice[1]-1-r), z_waypoints[j])]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_up += 1
                                prev_room = current_room
                    if 'hallway' in current_room:
                        advice += 'Proceed to ' + 'hallway '
                    elif 'bridge' in current_room:
                        advice += 'Proceed to ' + 'bridge '
                    elif 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes_up == 0 and num_room_changes_down == 0:
                            advice += 'Proceed to the door immediately '
                    else:
                        if next_direction == 'S':
                            advice += 'Proceed to '
                            if num_room_changes_down == 1:
                                advice += 'first'
                            if num_room_changes_down == 2:
                                advice += 'second'
                            if num_room_changes_down == 3:
                                advice += 'third'
                            advice += ' door on your right. '
                        if next_direction == 'N':
                            advice += 'Proceed to '
                            if num_room_changes_up == 1:
                                advice += 'first'
                            if num_room_changes_up == 2:
                                advice += 'second'
                            if num_room_changes_up == 3:
                                advice += 'third'
                            advice += ' door on your left. '
                        if next_direction == 'E':
                            advice += 'Turn around. '
                            # advice += 'behind you. '
                if current_direction == 'E':
                    num_room_changes_up = 0
                    num_room_changes_down = 0
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j] - (current_advice[1] - 1 - r)) ** 2 + (
                                              c[1] - z_waypoints[j]+10) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j]-(current_advice[1]-1-r), z_waypoints[j]-(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_down += 1
                                prev_room = current_room
                    prev_room = None
                    for r in range(current_advice[1]):
                        nearest = min(list(reversed_complete_room_dict.keys()),
                                      key=lambda c: (c[0] - x_waypoints[j] - (current_advice[1] - 1 - r)) ** 2 + (
                                              c[1] - z_waypoints[j] - 10) ** 2)
                        current_room = reversed_complete_room_dict[nearest]
                        # current_room = reversed_complete_room_dict[(x_waypoints[j]-(current_advice[1]-1-r), z_waypoints[j]-(current_advice[1]-1-r))]
                        if prev_room is None:
                            prev_room = current_room
                        else:
                            if prev_room != current_room and current_room != 'unidentified':
                                num_room_changes_up += 1
                                prev_room = current_room
                    if 'hallway' in current_room:
                        advice += 'Proceed to ' + 'hallway '
                    elif 'bridge' in current_room:
                        advice += 'Proceed to ' + 'bridge '
                    elif 'hallway' not in current_room and 'bridge' not in current_room:
                        if num_room_changes_up == 0 and num_room_changes_down == 0:
                            advice += 'Proceed to the door immediately '
                    else:
                        if next_direction == 'S':
                            advice += 'Proceed to '
                            if num_room_changes_down == 1:
                                advice += 'first'
                            if num_room_changes_down == 2:
                                advice += 'second'
                            if num_room_changes_down == 3:
                                advice += 'third'
                            advice += ' door on your right. '
                        if next_direction == 'N':
                            advice += 'Proceed to '
                            if num_room_changes_up == 1:
                                advice += 'first'
                            if num_room_changes_up == 2:
                                advice += 'second'
                            if num_room_changes_up == 3:
                                advice += 'third'
                            advice += ' door on your left. '
                        if next_direction == 'W':
                            advice += 'Turn around. '
                            # advice += 'behind you. '

            # advice += 'Orient ' + str(next_direction) + '. '

            if next_direction == 'N' or next_direction == 'S':
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))

            current_direction = next_direction
        prev_loc = (x_waypoints[j], z_waypoints[j])
    # if current_advice[1] > 0:
    #     advice += current_advice[0] + str(current_advice[1]) + ' steps. '
    if (x_waypoints[-1], z_waypoints[-1]) in yellow_locs or (x_waypoints[-1], z_waypoints[-1]) in green_locs:
        advice += 'Triage Victim.'
    # final_advice = ' '.join([x+'. ' for x in advice.split('. ')[:2]])
    final_advice = advice
    return final_advice


def convert_path_to_level_2_new(prev_loc, travel_path, gameboard, obstacles, yellow_locs, green_locs, doors,
                            stairs, goal_tuple_to_id, id_to_goal_tuple, reversed_complete_room_dict,
                            decision_points, intersections, entrances):
    # decision_points = decision_point_dict()
    loc_tuples = travel_path
    x_waypoints = [loc[0] for loc in loc_tuples]
    z_waypoints = [loc[1] for loc in loc_tuples]

    intersection_segments = {}
    current_path = []
    for (i,j) in travel_path:
        current_path.append((i, j))
        for intersection_key in intersections:
            if euclidean((i,j), intersections[intersection_key]['location']) <= 3:
                intersection_segments[intersection_key] = current_path
                current_path = []





    return ''


def convert_path_to_level_3(prev_loc, final_dest, gameboard, obstacles, yellow_locs, green_locs, doors, stairs,
                            goal_tuple_to_id, id_to_goal_tuple, reversed_complete_room_dict):

    room_to_go = reversed_complete_room_dict[(final_dest[0], final_dest[1])]
    advice = ''
    advice += 'In '+room_to_go + ', '
    if (final_dest[1], final_dest[0]) in yellow_locs:
        advice += 'save Yellow Victim.'

    if (final_dest[1], final_dest[0]) in green_locs:
        advice += 'save Green Victim.'

    final_advice = advice
    return final_advice

def get_path(victim_path, id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs):
    all_paths = {}
    prev_loc = (6,5)

    for i in victim_path:
        print('victim_path', id_to_goal_tuple[i])
        single_search = Search(gameboard, (6, 5), (6, 5), obstacles, yellow_locs, green_locs)
        travel_path = single_search.a_star_new(prev_loc, id_to_goal_tuple[i])
        all_paths[(prev_loc, id_to_goal_tuple[i])] = travel_path
        prev_loc = id_to_goal_tuple[i]


def generate_level_3(player, goal_to_goal_instruction_level_3, reversed_complete_room_dict, position_to_room_dict, current_index):
    next_goal = len(player.saved)
    coordinate = (int(player.x/10)+1, int(player.y/10)+1)

    nearest = min(list(reversed_complete_room_dict.keys()), key=lambda c: (c[0] - coordinate[0]) ** 2 + (c[1] - coordinate[1]) ** 2)
    # nearest = (nearest[0], nearest[1])
    # print("nearest in list", nearest in list(reversed_complete_room_dict.keys()))

    next_advice = goal_to_goal_instruction_level_3[(next_goal-1, next_goal)]['advice']
    give_advice = next_advice

    # print(nearest)
    # print(coordinate)
    # print(reversed_complete_room_dict[nearest])
    # print(goal_to_goal_instruction_level_3[(next_goal-1, next_goal)]['room'])

    if reversed_complete_room_dict[nearest] == goal_to_goal_instruction_level_3[(next_goal-1, next_goal)]['room']:
        if '. ' in next_advice:
            give_advice = next_advice.split('. ')[1]
        # else:
        #     give_advice = next_advice
    else:
        if '. ' in next_advice:
            give_advice = (next_advice.split('. ')[0])+'.'
        else:
            give_advice = next_advice

    return give_advice, current_index

def instructions_for_level_3(gameboard, id_to_goal_tuple, yellow_locs):
    with open('setup_files/aggregate_path_22vic_2.pkl', 'rb') as file:
        victim_path = pkl.load(file)
    goal_to_goal_instruction = {
    }

    rooms_dictionary, position_to_room_dict, complete_room_dict, reversed_complete_room_dict = get_rooms_dict(gameboard)

    previous_room = 'Start'
    for idx in range(len(victim_path)):
        advice = ''
        i = victim_path[idx]
        loc = id_to_goal_tuple[i]
        newkey = str(loc[0]) + ',' + str(loc[1])
        if newkey not in position_to_room_dict:
            room_to_go = reversed_complete_room_dict[loc]
        else:
            room_to_go = position_to_room_dict[newkey]

        if room_to_go == previous_room or room_to_go == 'unidentified':
            pass
        else:
            previous_room = room_to_go
            advice += (' Go to ' + room_to_go + ".")

        if (id_to_goal_tuple[i][1], id_to_goal_tuple[i][0]) in yellow_locs:
            advice += (' Save Yellow Victim.')
        else:
            advice += (' Save Green Victim.')

        goal_to_goal_instruction[(idx-1, idx)] = {}
        goal_to_goal_instruction[(idx-1, idx)]['advice'] = advice
        goal_to_goal_instruction[(idx - 1, idx)]['room'] = room_to_go


    return goal_to_goal_instruction

def generate_level_2(player, goal_to_goal_instruction_level_2, reversed_complete_room_dict, position_to_room_dict, current_index):
    next_goal = len(player.saved)
    current_advice_list = goal_to_goal_instruction_level_2[(next_goal-1, next_goal)]
    coordinate = (int(player.x / 10), int(player.y / 10))
    if abs(coordinate[0] - goal_to_goal_instruction_level_2[(next_goal-1, next_goal)][current_index]['target'][0])<2 and \
            abs(coordinate[1] - goal_to_goal_instruction_level_2[(next_goal-1, next_goal)][current_index]['target'][1])<2:
        current_index += 1
    if current_index not in current_advice_list:
        current_index -= 1
    curr_advice = current_advice_list[current_index]['instruction']
    return curr_advice, current_index

def instructions_for_level_2():
    goal_to_goal_instruction = {
        (-1, 0): {
            0: {
                'target': (13, 4),
                'instruction': 'Proceed to door.',
                },
            1: {
                'target': (17,4),
                'instruction': 'Enter room.',
            },
            2: {
                'target': (20,7),
                'instruction': 'Proceed to hallway on your right.',
            },
            3: {
                'target': (30,7),
                'instruction': 'Orient East and proceed to the first open doorway on left.',
            },
            4: {
                'target': (30,3),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (0, 1): {
            0: {
                'target': (32,6),
                'instruction': 'Proceed to bottom-right corner of room.',
            },
            1: {
                'target': (34,7),
                'instruction': 'Approach and triage yellow victim.',
            },
        },
        (1,2): {
            0: {
                'target': (30,10),
                'instruction': 'Exit room and enter hallway.',
            },
            1: {
                'target': (25,12),
                'instruction': 'Orient West and Proceed to first door on your left.',
            },
            2: {
                'target': (25,23),
                'instruction': 'Enter room and proceed to the center of room.',
            },
            3: {
                'target': (23,26),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (2,3): {
            0: {
                'target': (30, 33),
                'instruction': 'Proceed to bottom-right corner of room.',
            },
            1: {
                'target': (31,31),
                'instruction': 'Approach and triage yellow victim.',
            },
        },
        (3,4): {
            0: {
                'target': (24,36),
                'instruction': 'Proceed to South door and Exit room.',
            },
            1: {
                'target': (22,39),
                'instruction': 'Orient West and Proceed to first door on your left.',
            },
            2: {
                'target': (22,42),
                'instruction': 'Enter room.',
            },
            3: {
                'target': (26,42),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (4,5): {
            0: {
                'target': (22,40),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (20,38),
                'instruction': 'Orient West.',
            },
            2: {
                'target': (12,38),
                'instruction': 'Proceed to first door on your left.',
            },
            3: {
                'target': (13,42),
                'instruction': 'Enter room.',
            },
            4: {
                'target': (19,48),
                'instruction': 'Approach and triage yellow victim.',
            },
        },
        (5,6): {
            0: {
                'target': (13,38),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (40,38),
                'instruction': 'Orient East and Proceed forward to third door on your right.',
            },
            2: {
                'target': (45,48),
                'instruction': 'Approach and triage yellow victim',
            },
        },
        (6,7): {
            0: {
                'target': (40,38),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (59,37),
                'instruction': 'Orient East and Proceed to first hallway on your left. ',
            },
            2: {
                'target': (61,32),
                'instruction': 'Enter hallway and enter first room on your right.',
            },
            3: {
                'target': (64,32),
                'instruction': 'Approach and triage yellow victim.',
            },
        },
        (7,8): {
            0: {
                'target': (66, 19),
                'instruction': 'Orient North and Approach and triage yellow victim.',
            },
        },
        (8,9): {
            0: {
                'target': (69,15),
                'instruction': 'Proceed to north-side door.',
            },
            1: {
                'target': (71,15),
                'instruction': 'Exit room.',
            },
            2: {
                'target': (75,23),
                'instruction': 'Orient South and Proceed to first door on your left and enter room.',
            },
            3: {
                'target': (78,22),
                'instruction': 'Approach and triage yellow victim.',
            },
        },
        (9,10): {
            0: {
                'target': (74,24),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (73,11),
                'instruction': 'Orient North and Proceed to end of hallway.',
            },
            2: {
                'target': (69,11),
                'instruction': 'Orient West and enter hallway.',
            },
            3: {
                'target': (59,11),
                'instruction': 'Proceed forward to first hallway on your left.',
            },
            4: {
                'target': (57,23),
                'instruction': 'Orient South and Proceed to first door on right.',
            },
            5: {
                'target': (42,22),
                'instruction': 'Approach and triage yellow victim.',
            },
        },
        (10,11): {
            0: {
                'target': (58,22),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (57,25),
                'instruction': 'Orient South and Proceed to first door immediately on your right. ',
            },
            2: {
                'target': (54,25),
                'instruction': 'Enter room.',
            },
            3: {
                'target': (41,35),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (11,12): {
            0: {
                'target': (59,25),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (59,39),
                'instruction': 'Orient South and Proceed to door at the end of the hallway. ',
            },
            2: {
                'target': (59,41),
                'instruction': 'Enter room.',
            },
            3: {
                'target': (62,48),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (12,13): {
            0: {
                'target': (59,39),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (77,39),
                'instruction': 'Orient East and Proceed to second door on your right.',
            },
            2: {
                'target': (77,42),
                'instruction': 'Enter room.',
            },
            3: {
                'target': (80,48),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (13,14): {
            0: {
                'target': (77,40),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (77,36),
                'instruction': 'Orient North and Proceed to first door across the hallway.',
            },
            2: {
                'target': (77, 34),
                'instruction': 'Enter room.',
            },
            3: {
                'target': (82,35),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (14,15): {
            0: {
                'target': (77,36),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (74,37),
                'instruction': 'Orient west and enter hallway on your right.',
            },
            2: {
                'target': (73,10),
                'instruction': 'Proceed to end of hallway. ',
            },
            3: {
                'target': (70,8),
                'instruction': 'Orient West and proceed to first door on your right.',
            },
            4: {
                'target': (70,6),
                'instruction': 'Enter room.',
            },
            5: {
                'target': (77,6),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (15,16): {
            0: {
                'target': (83,7),
                'instruction': 'Orient East and Proceed to door at center of room.',
            },
            1: {
                'target': (85,7),
                'instruction': 'Enter room.',
            },
            2: {
                'target': (86,4),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (16,17): {
            0: {
                'target': (83,8),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (71,8),
                'instruction': 'Exit room.',
            },
            2: {
                'target': (38,10),
                'instruction': 'Orient West and Proceed to second door on your right.',
            },
            3: {
                'target': (38,8),
                'instruction': 'Enter room.',
            },
            4: {
                'target': (49,6),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (17,18): {
            0: {
                'target': (83, 8),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (18,19): {
            0: {
                'target': (36,7),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (38,17),
                'instruction': 'Orient South and Proceed across hallway to first door on your left',
            },
            2: {
                'target': (41,17),
                'instruction': 'Enter Room.',
            },
            3: {
                'target': (44,21),
                'instruction': 'Enter first stall.',
            },
            4: {
                'target': (41,21),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (19,20): {
            0: {
                'target': (38,18),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (38,26),
                'instruction': 'Orient South and Proceed first door immediately on your left.',
            },
            2: {
                'target': (41,26),
                'instruction': 'Enter Room.',
            },
            3: {
                'target': (45,34),
                'instruction': 'Enter second stall.',
            },
            4: {
                'target': (40,34),
                'instruction': 'Approach and triage green victim.',
            },
        },
        (20,21): {
            0: {
                'target': (38,27),
                'instruction': 'Exit room.',
            },
            1: {
                'target': (36,38),
                'instruction': 'Orient South and Proceed end of hallway.',
            },
            2: {
                'target': (31,39),
                'instruction': 'Orient West and Proceed to first door immediately on your left.',
            },
            3: {
                'target': (31,42),
                'instruction': 'Enter room.',
            },
            4: {
                'target': (37,48),
                'instruction': 'Approach and triage green victim.',
            },
        },
    }
    return goal_to_goal_instruction

def generate_level_1(player, goal_to_goal_instruction_level_1, reversed_complete_room_dict, position_to_room_dict, current_index):
    next_goal = len(player.saved)
    current_advice_list = goal_to_goal_instruction_level_1[(next_goal - 1, next_goal)]
    coordinate = (int(player.x / 10), int(player.y / 10))
    if abs(coordinate[0] - goal_to_goal_instruction_level_1[(next_goal - 1, next_goal)][current_index]['target'][0]) < 1 and \
            abs(coordinate[1] - goal_to_goal_instruction_level_1[(next_goal - 1, next_goal)][current_index]['target'][1]) < 1:
        current_index += 1
    if current_index not in current_advice_list:
        current_index -= 1
    curr_advice = current_advice_list[current_index]['instruction']
    return curr_advice, current_index

def instructions_for_level_1():
    gameboard, obstacles, yellow_locs, green_locs, doors, stairs = get_inv_gameboard()
    with open('setup_files/all_paths_22vic_2.pkl', 'rb') as file:
        all_paths = pkl.load(file)
    with open('setup_files/aggregate_path_22vic_2.pkl', 'rb') as file:
        victim_path = pkl.load(file)
    goal_tuple_to_id, goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix = load_goal_distances(cutoff_victims=34)
    rooms_dictionary, position_to_room_dict, complete_room_dict, reversed_complete_room_dict = get_rooms_dict(gameboard)
    goal_to_goal_instruction = {
    }
    total = 0
    cur_color = 'r'
    prev_loc = (6,5)
    current_advice = ('Walk forward ', 0)
    current_direction = 'N'
    for idx in range(len(victim_path)):
        advice = {}
        counter = 0
        i = victim_path[idx]
        loc_tuples  = all_paths[(prev_loc, id_to_goal_tuple[i])]
        x_waypoints = [loc[0] for loc in loc_tuples]
        z_waypoints = [loc[1] for loc in loc_tuples]

        prev_loc = (x_waypoints[1], z_waypoints[1])
        for j in range(2, len(x_waypoints)):
            if x_waypoints[j] < prev_loc[0]:
                next_direction = 'W'
            elif x_waypoints[j] > prev_loc[0]:
                next_direction = 'E'
            elif z_waypoints[j] < prev_loc[1]:
                next_direction = 'N'
            elif z_waypoints[j] > prev_loc[1]:
                next_direction = 'S'
            # else:
            #     next_direction = current_direction
            if next_direction == current_direction:
                if next_direction == 'N' or next_direction == 'S':
                    current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
                else:
                    current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))
            else:
                if current_advice[1] > 0:
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[j-1], z_waypoints[j-1])
                    advice[counter]['instruction'] = current_advice[0]+ str(current_advice[1]) + ' steps. '
                    counter += 1

                if current_direction == 'E' and next_direction == 'N':
                    add_advice = 'Turn Left. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'W' and next_direction == 'N':
                    add_advice = 'Turn Right. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'S' and next_direction == 'N':
                    add_advice = 'Turn Around. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1


                if current_direction == 'E' and next_direction == 'S':
                    add_advice = 'Turn Right. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'W' and next_direction == 'S':
                    add_advice = 'Turn Left. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'N' and next_direction == 'S':
                    add_advice = 'Turn Around. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'N' and next_direction == 'E':
                    add_advice = 'Turn Right. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'W' and next_direction == 'E':
                    add_advice = 'Turn Around. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'S' and next_direction == 'E':
                    add_advice = 'Turn Left. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'N' and next_direction == 'W':
                    add_advice = 'Turn Left. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'E' and next_direction == 'W':
                    add_advice = 'Turn Around. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if current_direction == 'S' and next_direction == 'W':
                    add_advice = 'Turn Right. '
                    advice[counter] = {}
                    advice[counter]['target'] = (x_waypoints[min(len(x_waypoints)-1, j+1)], z_waypoints[min(len(z_waypoints)-1, j+1)])
                    advice[counter]['instruction'] = add_advice
                    counter += 1

                if next_direction == 'N' or next_direction == 'S':
                    current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))

                else:
                    current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))


                current_direction = next_direction
            prev_loc = (x_waypoints[j], z_waypoints[j])

        if current_advice[1] > 0:
            advice[counter] = {}
            advice[counter]['target'] = (x_waypoints[-1], z_waypoints[-1])
            advice[counter]['instruction'] = current_advice[0] + str(current_advice[1]) + ' steps. Swing Medical Ax 10 Times.'
            counter += 1

        # advice[counter] = {}
        # advice[counter]['target'] = (x_waypoints[-1], z_waypoints[-1])
        # advice[counter]['instruction'] = 'Swing Medical Ax 10 Times.'
        # counter += 1

        goal_to_goal_instruction[(idx-1, idx)] = advice
        prev_loc = id_to_goal_tuple[i]
        current_advice = ('Walk forward ', 0)

    return goal_to_goal_instruction

def generate_level_2_astar_decision_points_orig(start_idx, goal_idx, decision_points, decision_neighbors,
                                           reversed_complete_room_dict, victim_idx, curr_coordinate, id_to_goal_tuple):

    decision_search = Decision_Point_Search(decision_neighbors, decision_points)
    path = decision_search.a_star(start_idx, goal_idx)

    if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, curr_coordinate)] == reversed_complete_room_dict[id_to_goal_tuple[victim_idx]]:
        return 'Approach and triage victim. '


    if len(path) < 2:
        return 'Enter room. Approach and triage victim. '

    advice = []
    # curr_direction = 'North'
    # if decision_points[path[0]]['type'] == 'door' and decision_points[path[1]]['type'] == 'entrance':
    #     advice.append('Exit room. ')
    #     if decision_points[path[1]]['location'][0]+2 < decision_points[path[2]]['location'][0]:
    #         curr_direction = 'East'
    #     if decision_points[path[1]]['location'][0] > decision_points[path[2]]['location'][0]+2:
    #         curr_direction = 'West'
    #     if decision_points[path[1]]['location'][1]+2 < decision_points[path[2]]['location'][1]:
    #         curr_direction = 'South'
    #     if decision_points[path[1]]['location'][1] > decision_points[path[2]]['location'][1]+2:
    #         curr_direction = 'North'

    # print('path', path)
    rooms_on_left = []
    rooms_on_right = []
    room_direction = None
    for i in range(0, len(path)):
        waypt_idx = path[i]
        if decision_points[waypt_idx]['type'] == 'door':
            if decision_points[path[i-1]]['type'] == 'door':
                advice.append('Cross room and proceed to other door. ')
                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and exit room. ')
            else:
                if len(rooms_on_left) > 0:
                    advice.append('Pass '+str(len(rooms_on_left)) + ' doors on left. ')
                if len(rooms_on_right) > 0:
                    advice.append('Pass '+str(len(rooms_on_right)) + ' doors on right. ')

                if room_direction == 'Left':
                    advice.append('Proceed to no.'+str(len(rooms_on_left)+1) + ' door on left. ')
                if room_direction == 'Right':
                    advice.append('Proceed to no.'+str(len(rooms_on_right)+1) + ' door on right. ')

                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and enter room. ')

        if decision_points[waypt_idx]['type'] == 'intersection':
            if len(rooms_on_left) > 0:
                advice.append('Pass '+str(len(rooms_on_left)) + ' doors on left. ')
            if len(rooms_on_right) > 0:
                advice.append('Pass '+str(len(rooms_on_right)) + ' doors on right. ')
            rooms_on_left = []
            rooms_on_right = []

            if decision_points[path[i+1]]['location'][0]+2 < decision_points[path[i]]['location'][0]:
                if curr_direction == 'North':
                    curr_direction = 'West'
                elif curr_direction == 'South':
                    curr_direction = 'East'
            elif decision_points[path[i+1]]['location'][0] > decision_points[path[i]]['location'][0]+2:
                if curr_direction == 'North':
                    curr_direction = 'East'
                elif curr_direction == 'South':
                    curr_direction = 'West'
            elif decision_points[path[i+1]]['location'][1]+2 < decision_points[path[i]]['location'][1]:
                if curr_direction == 'East':
                    curr_direction = 'North'
                elif curr_direction == 'West':
                    curr_direction = 'South'
            elif decision_points[path[i+1]]['location'][1] > decision_points[path[i]]['location'][1]+2:
                if curr_direction == 'East':
                    curr_direction = 'South'
                elif curr_direction == 'West':
                    curr_direction = 'North'
            advice.append('Proceed to intersection, and orient '+ curr_direction +'. ')

        # room_direction = None
        if decision_points[waypt_idx]['type'] == 'entrance':

            paired_door = decision_points[waypt_idx]['paired_doors'][0]
            if decision_points[paired_door]['location'][0] < decision_points[waypt_idx]['location'][0]:
                if curr_direction == 'North':
                    room_direction = 'Left'
                if curr_direction == 'South':
                    room_direction = 'Right'
                if curr_direction == 'East':
                    room_direction = 'Ahead'
                if curr_direction == 'West':
                    room_direction = 'Ahead'
            elif decision_points[paired_door]['location'][0] > decision_points[waypt_idx]['location'][0]:
                if curr_direction == 'North':
                    room_direction = 'Right'
                if curr_direction == 'South':
                    room_direction = 'Left'
                if curr_direction == 'East':
                    room_direction = 'Ahead'
                if curr_direction == 'West':
                    room_direction = 'Ahead'
            elif decision_points[paired_door]['location'][1] < decision_points[waypt_idx]['location'][1]:
                if curr_direction == 'North':
                    room_direction = 'Ahead'
                if curr_direction == 'South':
                    room_direction = 'Ahead'
                if curr_direction == 'East':
                    room_direction = 'Left'
                if curr_direction == 'West':
                    room_direction = 'Right'
            elif decision_points[paired_door]['location'][1] > decision_points[waypt_idx]['location'][1]:
                if curr_direction == 'North':
                    room_direction = 'Ahead'
                if curr_direction == 'South':
                    room_direction = 'Ahead'
                if curr_direction == 'East':
                    room_direction = 'Right'
                if curr_direction == 'West':
                    room_direction = 'Left'

            if room_direction == 'Left':
                rooms_on_left.append(room_direction)
            else:
                rooms_on_right.append(room_direction)

            if decision_points[path[i]]['location'][0]+2 < decision_points[path[i-1]]['location'][0]:
                curr_direction = 'West'
            if decision_points[path[i]]['location'][0] > decision_points[path[i-1]]['location'][0]+2:
                curr_direction = 'East'
            if decision_points[path[i]]['location'][1]+2 < decision_points[path[i-1]]['location'][1]:
                curr_direction = 'North'
            if decision_points[path[i]]['location'][1] > decision_points[path[i-1]]['location'][1]+2:
                curr_direction = 'South'


        # if room_direction == None:
        #     advice.append(decision_points[waypt_idx]['type'])
        # else:
        #     advice.append(decision_points[waypt_idx]['type'] + room_direction)

        # print('advice', advice)
    generated_advice = ''
    for ad in advice:
        generated_advice += ad
    return generated_advice

def generate_level_2_astar_decision_points_orig_2(start_idx, goal_idx, decision_points, decision_neighbors,
                                           reversed_complete_room_dict, victim_idx, curr_coordinate, prev_coordinate, id_to_goal_tuple):

    decision_search = Decision_Point_Search(decision_neighbors, decision_points)
    path = decision_search.a_star(start_idx, goal_idx)
    # print('path = ', path)

    # if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, curr_coordinate)] == reversed_complete_room_dict[id_to_goal_tuple[victim_idx]]:
    #     return 'Approach and triage victim. '


    if len(path) < 2:
        return 'Enter room. Approach and triage victim. '

    advice = []
    # x_diff = curr_coordinate[0] - prev_coordinate[0]
    # z_diff = curr_coordinate[1] - prev_coordinate[1]
    x_diff = decision_points[path[1]]['location'][0] - curr_coordinate[0]
    z_diff = decision_points[path[1]]['location'][1] - curr_coordinate[1]
    if abs(x_diff) > abs(z_diff):
        if x_diff < 0:
            curr_direction = 'W'
        else:
            curr_direction = 'E'
    else:
        if z_diff < 0:
            curr_direction = 'N'
        else:
            curr_direction = 'S'
    # print('x_diff', x_diff)
    # print('z_diff', z_diff)
    advice.append('Head '+curr_direction+'. ')
    # print('path', path)
    rooms_on_left = []
    rooms_on_right = []
    room_direction = None
    for i in range(0, len(path)):
        waypt_idx = path[i]
        # prev_waypt_idx = path[i-1]
        # next_waypt_idx = path[i+1]

        # Set current direction (from previous point)
        if i > 0:
            prev_waypt_idx = path[i-1]
            x_diff = decision_points[path[i]]['location'][0] - decision_points[path[i-1]]['location'][0]
            z_diff = decision_points[path[i]]['location'][1] - decision_points[path[i-1]]['location'][1]
            if abs(x_diff) > abs(z_diff):
                if x_diff < 0:
                    curr_direction = 'W'
                else:
                    curr_direction = 'E'
            else:
                if z_diff < 0:
                    curr_direction = 'N'
                else:
                    curr_direction = 'S'



        if decision_points[waypt_idx]['type'] == 'door':
            if i > 0 and decision_points[path[i-1]]['type'] == 'entrance':
                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and enter room. ')

            elif i > 0 and decision_points[path[i-1]]['type'] == 'door':
                advice.append('Cross room and proceed to other door. ')
                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and exit room. ')

            else:
                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and proceed. ')

        if decision_points[waypt_idx]['type'] == 'intersection':
            if len(rooms_on_left) > 0:
                advice.append('Pass '+str(len(rooms_on_left)) + ' doors on left. ')
            if len(rooms_on_right) > 0:
                advice.append('Pass '+str(len(rooms_on_right)) + ' doors on right. ')
            rooms_on_left = []
            rooms_on_right = []

            if i < len(path)-1:
                # next_waypt_idx = path[i + 1]
                x_diff = decision_points[path[i+1]]['location'][0] - decision_points[path[i]]['location'][0]
                z_diff = decision_points[path[i+1]]['location'][1] - decision_points[path[i]]['location'][1]
                if abs(x_diff) > abs(z_diff):
                    if x_diff < 0:
                        next_direction = 'W'
                    else:
                        next_direction = 'E'
                else:
                    if z_diff < 0:
                        next_direction = 'N'
                    else:
                        next_direction = 'S'
                advice.append('Proceed to intersection, and orient ' + next_direction + '. ')
            else:
                advice.append('Proceed to intersection.' )


        # room_direction = None
        if decision_points[waypt_idx]['type'] == 'entrance':

            paired_door = decision_points[waypt_idx]['paired_doors'][0]

            x_diff = decision_points[paired_door]['location'][0] - decision_points[path[i]]['location'][0]
            z_diff = decision_points[paired_door]['location'][1] - decision_points[path[i]]['location'][1]
            if abs(x_diff) > abs(z_diff):
                if x_diff < 0:
                    if curr_direction == 'N':
                        room_direction = 'Left'
                    if curr_direction == 'S':
                        room_direction = 'Right'
                    if curr_direction == 'E':
                        room_direction = 'Ahead'
                    if curr_direction == 'W':
                        room_direction = 'Ahead'

                else:
                    if curr_direction == 'N':
                        room_direction = 'Right'
                    if curr_direction == 'S':
                        room_direction = 'Left'
                    if curr_direction == 'E':
                        room_direction = 'Ahead'
                    if curr_direction == 'W':
                        room_direction = 'Ahead'
            else:
                if z_diff < 0:
                    if curr_direction == 'N':
                        room_direction = 'Ahead'
                    if curr_direction == 'S':
                        room_direction = 'Ahead'
                    if curr_direction == 'E':
                        room_direction = 'Left'
                    if curr_direction == 'W':
                        room_direction = 'Right'
                else:
                    if curr_direction == 'N':
                        room_direction = 'Ahead'
                    if curr_direction == 'S':
                        room_direction = 'Ahead'
                    if curr_direction == 'E':
                        room_direction = 'Right'
                    if curr_direction == 'W':
                        room_direction = 'Left'

            if room_direction == 'Left':
                rooms_on_left.append(room_direction)
            elif room_direction == 'Right':
                rooms_on_right.append(room_direction)

            if i < len(path)-1:
                if decision_points[path[i+1]]['type'] == 'door':
                    # if len(rooms_on_left) > 0:
                    #     advice.append('Pass ' + str(len(rooms_on_left)) + ' doors on left. ')
                    # if len(rooms_on_right) > 0:
                    #     advice.append('Pass ' + str(len(rooms_on_right)) + ' doors on right. ')
                    if room_direction == 'Left':
                        if len(rooms_on_left) > 0:
                            advice.append('Proceed to no.' + str(len(rooms_on_left)) + ' door on left. ')
                        else:
                            advice.append('Proceed to the first door on left. ')
                    if room_direction == 'Right':
                        if len(rooms_on_right) > 0:
                            advice.append('Proceed to no.' + str(len(rooms_on_right)) + ' door on right. ')
                        else:
                            advice.append('Proceed to the first door on right. ')
                    rooms_on_left = []
                    rooms_on_right = []
                    # advice.append('Approach door on ' + room_direction + '. ')

        if i == len(path)-1 and decision_points[waypt_idx]['type'] == 'victim':
            advice.append('Approach and triage victim.')

    generated_advice = ''
    for ad in advice[:2]:
        generated_advice += ad
    return generated_advice

def generate_level_2_astar_decision_points_old(start_idx, goal_idx, decision_points, decision_neighbors,
                                           reversed_complete_room_dict, victim_idx, curr_coordinate, prev_coordinate, id_to_goal_tuple):

    decision_search = Decision_Point_Search(decision_neighbors, decision_points)
    path = decision_search.a_star(start_idx, goal_idx)
    # print('path = ', path)

    # if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, curr_coordinate)] == reversed_complete_room_dict[id_to_goal_tuple[victim_idx]]:
    #     return 'Approach and triage victim. '


    if len(path) < 2:
        return 'Enter room. Approach and triage victim. '

    advice = []
    # x_diff = curr_coordinate[0] - prev_coordinate[0]
    # z_diff = curr_coordinate[1] - prev_coordinate[1]
    x_diff = decision_points[path[1]]['location'][0] - curr_coordinate[0]
    z_diff = decision_points[path[1]]['location'][1] - curr_coordinate[1]
    if abs(x_diff) > abs(z_diff):
        if x_diff < 0:
            curr_direction = 'W'
        else:
            curr_direction = 'E'
    else:
        if z_diff < 0:
            curr_direction = 'N'
        else:
            curr_direction = 'S'
    # print('x_diff', x_diff)
    # print('z_diff', z_diff)
    advice.append('Head: '+curr_direction+'. ')
    # print('path', path)
    rooms_on_left = []
    rooms_on_right = []
    room_direction = None
    for i in range(0, len(path)):
        waypt_idx = path[i]
        # prev_waypt_idx = path[i-1]
        # next_waypt_idx = path[i+1]

        # Set current direction (from previous point)
        if i > 0:
            prev_waypt_idx = path[i-1]
            x_diff = decision_points[path[i]]['location'][0] - decision_points[path[i-1]]['location'][0]
            z_diff = decision_points[path[i]]['location'][1] - decision_points[path[i-1]]['location'][1]
            if abs(x_diff) > abs(z_diff):
                if x_diff < 0:
                    curr_direction = 'W'
                else:
                    curr_direction = 'E'
            else:
                if z_diff < 0:
                    curr_direction = 'N'
                else:
                    curr_direction = 'S'



        if decision_points[waypt_idx]['type'] == 'door':
            if i > 0 and decision_points[path[i-1]]['type'] == 'entrance':
                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and enter room. ')

            elif i > 0 and decision_points[path[i-1]]['type'] == 'door':
                advice.append('Cross room and proceed to other door. ')
                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and exit room. ')

            else:
                rooms_on_left = []
                rooms_on_right = []
                advice.append('Open door and proceed. ')

        if decision_points[waypt_idx]['type'] == 'intersection':
            if len(rooms_on_left) > 0:
                advice.append('Pass '+str(len(rooms_on_left)) + ' doors on left. ')
            if len(rooms_on_right) > 0:
                advice.append('Pass '+str(len(rooms_on_right)) + ' doors on right. ')
            rooms_on_left = []
            rooms_on_right = []

            if i < len(path)-1:
                # next_waypt_idx = path[i + 1]
                x_diff = decision_points[path[i+1]]['location'][0] - decision_points[path[i]]['location'][0]
                z_diff = decision_points[path[i+1]]['location'][1] - decision_points[path[i]]['location'][1]
                if abs(x_diff) > abs(z_diff):
                    if x_diff < 0:
                        next_direction = 'W'
                    else:
                        next_direction = 'E'
                else:
                    if z_diff < 0:
                        next_direction = 'N'
                    else:
                        next_direction = 'S'
                advice_to_add = 'Proceed to intersection'
                if next_direction != curr_direction:
                    advice_to_add += ', and orient '
                    advice_to_add += next_direction
                    advice_to_add += '. '
                else:
                    advice_to_add += '. '
            else:
                advice.append('Proceed to intersection.' )


        # room_direction = None
        if decision_points[waypt_idx]['type'] == 'entrance':

            paired_door = decision_points[waypt_idx]['paired_doors'][0]

            x_diff = decision_points[paired_door]['location'][0] - decision_points[path[i]]['location'][0]
            z_diff = decision_points[paired_door]['location'][1] - decision_points[path[i]]['location'][1]
            if abs(x_diff) > abs(z_diff):
                if x_diff < 0:
                    if curr_direction == 'N':
                        room_direction = 'Left'
                    if curr_direction == 'S':
                        room_direction = 'Right'
                    if curr_direction == 'E':
                        room_direction = 'Ahead'
                    if curr_direction == 'W':
                        room_direction = 'Ahead'

                else:
                    if curr_direction == 'N':
                        room_direction = 'Right'
                    if curr_direction == 'S':
                        room_direction = 'Left'
                    if curr_direction == 'E':
                        room_direction = 'Ahead'
                    if curr_direction == 'W':
                        room_direction = 'Ahead'
            else:
                if z_diff < 0:
                    if curr_direction == 'N':
                        room_direction = 'Ahead'
                    if curr_direction == 'S':
                        room_direction = 'Ahead'
                    if curr_direction == 'E':
                        room_direction = 'Left'
                    if curr_direction == 'W':
                        room_direction = 'Right'
                else:
                    if curr_direction == 'N':
                        room_direction = 'Ahead'
                    if curr_direction == 'S':
                        room_direction = 'Ahead'
                    if curr_direction == 'E':
                        room_direction = 'Right'
                    if curr_direction == 'W':
                        room_direction = 'Left'

            if room_direction == 'Left':
                rooms_on_left.append(room_direction)
            elif room_direction == 'Right':
                rooms_on_right.append(room_direction)

            if i < len(path)-1:
                if decision_points[path[i+1]]['type'] == 'door':
                    # if len(rooms_on_left) > 0:
                    #     advice.append('Pass ' + str(len(rooms_on_left)) + ' doors on left. ')
                    # if len(rooms_on_right) > 0:
                    #     advice.append('Pass ' + str(len(rooms_on_right)) + ' doors on right. ')
                    if room_direction == 'Left':
                        if len(rooms_on_left) > 0:
                            advice.append('Proceed to no.' + str(len(rooms_on_left)) + ' door on left. ')
                        else:
                            advice.append('Proceed to the first door on left. ')
                    if room_direction == 'Right':
                        if len(rooms_on_right) > 0:
                            advice.append('Proceed to no.' + str(len(rooms_on_right)) + ' door on right. ')
                        else:
                            advice.append('Proceed to the first door on right. ')
                    rooms_on_left = []
                    rooms_on_right = []
                    # advice.append('Approach door on ' + room_direction + '. ')

        if i == len(path)-1 and decision_points[waypt_idx]['type'] == 'victim':
            advice.append('Approach and triage victim.')

    generated_advice = ''
    for ad in advice[:2]:
        generated_advice += ad
    return generated_advice

def get_nearest_regular(dictionary, point):
    nearest = None
    min_dist = sys.maxsize
    for key in dictionary:
        d = euclidean(key, point)
        if d < min_dist:
            min_dist = d
            nearest = key
    return nearest


def get_nearest(dictionary, point):
    nearest = None
    min_dist = sys.maxsize
    for key in dictionary:
        d = euclidean(dictionary[key]['location'], point)
        if d < min_dist:
            min_dist = d
            nearest = key
    return nearest


def get_nearest_room_based(dictionary, point, reversed_complete_room_dict):
    rooms_to_decision_points_dictionary = rooms_to_decision_points()

    if point in reversed_complete_room_dict:
        # print('point in room: ', reversed_complete_room_dict[point])
        if reversed_complete_room_dict[point] in rooms_to_decision_points_dictionary:
            return rooms_to_decision_points_dictionary[reversed_complete_room_dict[point]]

    nearest = None
    min_dist = sys.maxsize
    for key in dictionary:
        d = euclidean(dictionary[key]['location'], point)
        if d < min_dist:
            min_dist = d
            nearest = key
    return nearest

def rooms_to_decision_points():
    rooms_to_decision_points_dictionary = {
        'Room 100': 0,
        'Open Break Area': 2,
        'Executive Suite 1': 3,
        'Executive Suite 2': 4,
        'King Chris\' Office': 5,
        # 'King Chris\' Office 2': 12,
        'Kings Terrace': 6,

        'Room 101': 27,
        'Room 102': 39,
        'Room 103': 57,

        'Room 104': 55,
        'Room 105': 53,
        'Room 106': 51,
        'Room 107': 49,
        'Room 108': 47,
        'Room 109': 45,
        'Room 110': 43,
        'Room 111': 41,

        'Herbalife Conference Room': 90,
        'Amway Conference Room': 23,
        'Mary Kay Conference Room': 25,
        'Women\'s Room': 29,
        'Men\'s Room': 17,
        'Den': 13,
        'The Computer Farm': 83,

    }
    return rooms_to_decision_points_dictionary


def euclidean_dist(x0, y0, x1, y1):
    return np.sqrt((x1-x0)**2 + (y1-y0)**2)

def manhattan_dist(x0, y0, x1, y1):
    return abs(x1-x0) + abs(y1-y0)


def check_approximate_direction(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    dir = 1 #1234 = NESW
    if abs(dx) > abs(dy):
        if dx < 0:
            # west, negative x change
            dir = 4
        else:
            dir = 2
    else:
        if dy < 0:
            # north, negative y change
            dir = 1
        else:
            dir = 3
    return dir, (dx, dy)

def generate_level_2_astar_decision_points_no_rotation(player, start_idx, goal_idx, decision_points, decision_neighbors,
                                           reversed_complete_room_dict, victim_idx, curr_coordinate, prev_coordinate, id_to_goal_tuple):

    decision_search = Decision_Point_Search(decision_neighbors, decision_points)
    path = decision_search.a_star(start_idx, goal_idx)
    # print('path = ', path)

    # if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, curr_coordinate)] == reversed_complete_room_dict[id_to_goal_tuple[victim_idx]]:
    #     return 'Approach and triage victim. '
    count_translation ={
        0: 'first',
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth'
    }
    direction_dict = {
        1: 'N',
        2: 'E',
        3: 'S',
        4: 'W',
    }
    color_dict = {
        'y' : 'yellow',
        'g' : 'green'
    }

    if len(path) <= 1:
        return 'Approach and triage ' + color_dict[decision_points[path[-1]]['color']] + ' victim. ', decision_points[len(path)-1]['location']

    advice = []
    need_to_reach_destinations = []
    start_direction = player.current_direction #1234 = NESW
    # Check if player is past the current start location
    if manhattan_dist(player.x, player.y, decision_points[path[1]]['location'][0], decision_points[path[1]]['location'][1]) <\
            manhattan_dist(decision_points[path[0]]['location'][0], decision_points[path[1]]['location'][1], decision_points[path[1]]['location'][0], decision_points[path[1]]['location'][1]):
        # player is closer to second goal point than first goal point
        start_decision_point_idx = 1
    else:
        # player is closer to first goal point
        start_decision_point_idx = 0

    if start_decision_point_idx == 0:
        current_direction, (dx, dy) = check_approximate_direction(player.x,
                                                     player.y,
                                                     decision_points[path[start_decision_point_idx]]['location'][0],
                                                     decision_points[path[start_decision_point_idx]]['location'][1])
        if dx + dy < 2:
            current_direction, (dx, dy) = check_approximate_direction(
                decision_points[path[start_decision_point_idx]]['location'][0],
                decision_points[path[start_decision_point_idx]]['location'][1],
                decision_points[path[start_decision_point_idx+1]]['location'][0],
                decision_points[path[start_decision_point_idx+1]]['location'][1])

    else:
        current_direction, (dx, dy) = check_approximate_direction(decision_points[path[start_decision_point_idx - 1]]['location'][0],
                                                     decision_points[path[start_decision_point_idx - 1]]['location'][1],
                                                     decision_points[path[start_decision_point_idx]]['location'][0],
                                                     decision_points[path[start_decision_point_idx]]['location'][1])




    if current_direction != start_direction and (dx+dy) > 2:
        advice.append("Head "+ direction_dict[current_direction] +'. ')
        need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])


    left_doors = []
    right_doors = []
    other_doors = []
    intersection_count = 0
    prev_direction = start_direction

    for index in range(start_decision_point_idx, len(path)-1):
        # check direction of current decision point from current location

        next_direction, (dx, dy) = check_approximate_direction(decision_points[path[index]]['location'][0],
                                                     decision_points[path[index]]['location'][1],
                                                     decision_points[path[index + 1]]['location'][0],
                                                     decision_points[path[index + 1]]['location'][1])


        if decision_points[path[index]]['type'] == 'entrance':
            paired_door = decision_points[path[index]]['paired_doors'][0]
            door_direction_absolute, (dx, dy) = check_approximate_direction(decision_points[path[index]]['location'][0], decision_points[path[index]]['location'][1],
                                                         decision_points[paired_door]['location'][0], decision_points[paired_door]['location'][1])

            # check if you just exited a room
            if index > 0 and decision_points[path[index-1]]['type'] == 'door':
                advice.append('Exit room and head ' + direction_dict[next_direction] + '. ')
                need_to_reach_destinations.append(decision_points[path[index]]['location'])
            else:
                # L,R = 0,1
                # Ahead, Behind = 2,3

                if current_direction == 1: # north
                    if door_direction_absolute == 1:
                        door_dir = 2
                    if door_direction_absolute == 2:
                        door_dir = 1
                    if door_direction_absolute == 3:
                        door_dir = 3
                    if door_direction_absolute == 4:
                        door_dir = 0
                if current_direction == 2: # east
                    if door_direction_absolute == 1:
                        door_dir = 0
                    if door_direction_absolute == 2:
                        door_dir = 2
                    if door_direction_absolute == 3:
                        door_dir = 1
                    if door_direction_absolute == 4:
                        door_dir = 3
                if current_direction == 3: # south
                    if door_direction_absolute == 1:
                        door_dir = 3
                    if door_direction_absolute == 2:
                        door_dir = 0
                    if door_direction_absolute == 3:
                        door_dir = 2
                    if door_direction_absolute == 4:
                        door_dir = 1
                if current_direction == 4: # west
                    if door_direction_absolute == 1:
                        door_dir = 1
                    if door_direction_absolute == 2:
                        door_dir = 3
                    if door_direction_absolute == 3:
                        door_dir = 0
                    if door_direction_absolute == 4:
                        door_dir = 2


                if door_dir == 0:
                    left_doors.append(paired_door)
                elif door_dir == 1:
                    right_doors.append(paired_door)
                else:
                    other_doors.append(paired_door)

                if next_direction != current_direction:
                    print("current_direction, next_direction: ", (current_direction, next_direction))
                    if current_direction == 1:
                        if next_direction == 4:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 2:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    if current_direction == 2:
                        if next_direction == 1:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 3:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    if current_direction == 3:
                        if next_direction == 2:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 4:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    if current_direction == 4:
                        if next_direction == 3:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 1:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])

                    right_doors = []
                    left_doors = []

        if decision_points[path[index]]['type'] == 'door':
            if index > 0 and decision_points[path[index-1]]['type'] == 'entrance':
                advice.append('Proceed through door. ')
                need_to_reach_destinations.append(decision_points[path[index]]['location'])

        if decision_points[path[index]]['type'] == 'intersection':
            # print("get to loc: ", path[index])
            # print("whole path ", [path[c] for c in range(len(path))])
            if index > 0:
                intersection_count += 1
                if next_direction != current_direction:
                    # if intersection_count > 0:
                    #     print("intersection count", intersection_count)
                    #     print('prev_direction', prev_direction)
                    #     print('current_direction', current_direction)
                    #     print("next_direction", next_direction)
                    #     print()
                    #

                    advice.append('Proceed to ' + count_translation[intersection_count] + ' intersection.')
                    advice.append("Head " + direction_dict[next_direction])
                    need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    intersection_count = 0
                    right_doors = []
                    left_doors = []

        prev_direction = current_direction
        current_direction = next_direction

    if decision_points[path[-1]]['type'] == 'victim':
        advice.append('Approach and triage ' + color_dict[decision_points[path[-1]]['color']] + ' victim.')
        need_to_reach_destinations.append(decision_points[path[-1]]['location'])

    generated_advice = ''

    for ad in advice[:2]:
        generated_advice += ad

    if len(advice) > 2:
        final_destination = need_to_reach_destinations[1]
    else:
        final_destination = decision_points[len(path)-1]['location']

    # for ad in advice:
    #     generated_advice += ad
    #
    # final_destination = decision_points[len(path)-1]['location']
    # print('advice', advice)
    return generated_advice, final_destination

def generate_level_2_astar_decision_points_w_rotation(player, start_idx, goal_idx, decision_points, decision_neighbors,
                                           reversed_complete_room_dict, victim_idx, curr_coordinate, prev_coordinate, id_to_goal_tuple):

    decision_search = Decision_Point_Search(decision_neighbors, decision_points)
    path = decision_search.a_star(start_idx, goal_idx)
    # print('path = ', path)

    # if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, curr_coordinate)] == reversed_complete_room_dict[id_to_goal_tuple[victim_idx]]:
    #     return 'Approach and triage victim. '
    count_translation ={
        0: 'first',
        1: 'first',
        2: 'second',
        3: 'third',
        4: 'fourth',
        5: 'fifth'
    }
    direction_dict = {
        1: 'N',
        2: 'E',
        3: 'S',
        4: 'W',
    }
    color_dict = {
        'y' : 'yellow',
        'g' : 'green'
    }

    if len(path) <= 1:
        return 'Find and triage ' + color_dict[decision_points[path[-1]]['color']] + ' victim. ', decision_points[len(path)-1]['location']

    advice = []
    need_to_reach_destinations = []
    start_direction = player.current_direction #1234 = NESW
    # Check if player is past the current start location
    if manhattan_dist(player.x, player.y, decision_points[path[1]]['location'][0], decision_points[path[1]]['location'][1]) <\
            manhattan_dist(decision_points[path[0]]['location'][0], decision_points[path[1]]['location'][1], decision_points[path[1]]['location'][0], decision_points[path[1]]['location'][1]):
        # player is closer to second goal point than first goal point
        start_decision_point_idx = 1
    else:
        # player is closer to first goal point
        start_decision_point_idx = 0

    if start_decision_point_idx == 0:
        current_direction, (dx, dy) = check_approximate_direction(player.x,
                                                     player.y,
                                                     decision_points[path[start_decision_point_idx]]['location'][0],
                                                     decision_points[path[start_decision_point_idx]]['location'][1])
        if dx + dy < 2:
            current_direction, (dx, dy) = check_approximate_direction(
                decision_points[path[start_decision_point_idx]]['location'][0],
                decision_points[path[start_decision_point_idx]]['location'][1],
                decision_points[path[start_decision_point_idx+1]]['location'][0],
                decision_points[path[start_decision_point_idx+1]]['location'][1])

    else:
        current_direction, (dx, dy) = check_approximate_direction(decision_points[path[start_decision_point_idx - 1]]['location'][0],
                                                     decision_points[path[start_decision_point_idx - 1]]['location'][1],
                                                     decision_points[path[start_decision_point_idx]]['location'][0],
                                                     decision_points[path[start_decision_point_idx]]['location'][1])



    if current_direction != start_direction:
        if start_direction == 1:
            if current_direction == 4:
                advice.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            elif current_direction == 2:
                advice.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            else:
                advice.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
        if start_direction == 2:
            if current_direction == 1:
                advice.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            elif current_direction == 3:
                advice.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            else:
                advice.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
        if start_direction == 3:
            if current_direction == 2:
                advice.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            elif current_direction == 4:
                advice.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            else:
                advice.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
        if start_direction == 4:
            if current_direction == 3:
                advice.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            elif current_direction == 1:
                advice.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])
            else:
                advice.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[path[start_decision_point_idx]]['location'])

    left_doors = []
    right_doors = []
    other_doors = []
    intersection_count = 0
    prev_direction = start_direction

    for index in range(start_decision_point_idx, len(path)-1):
        # check direction of current decision point from current location

        next_direction, (dx, dy) = check_approximate_direction(decision_points[path[index]]['location'][0],
                                                     decision_points[path[index]]['location'][1],
                                                     decision_points[path[index + 1]]['location'][0],
                                                     decision_points[path[index + 1]]['location'][1])


        if decision_points[path[index]]['type'] == 'entrance':
            paired_door = decision_points[path[index]]['paired_doors'][0]
            door_direction_absolute, (dx, dy) = check_approximate_direction(decision_points[path[index]]['location'][0], decision_points[path[index]]['location'][1],
                                                         decision_points[paired_door]['location'][0], decision_points[paired_door]['location'][1])

            # check if you just exited a room
            if index > 0 and decision_points[path[index-1]]['type'] == 'door':
                advice.append('Exit room. ')
                need_to_reach_destinations.append(decision_points[path[index]]['location'])
                if next_direction != current_direction:
                    if current_direction == 1:
                        if next_direction == 4:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 2:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                    if current_direction == 2:
                        if next_direction == 1:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 3:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                    if current_direction == 3:
                        if next_direction == 2:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 4:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                    if current_direction == 4:
                        if next_direction == 3:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 1:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
            else:
                # L,R = 0,1
                # Ahead, Behind = 2,3

                if current_direction == 1: # north
                    if door_direction_absolute == 1:
                        door_dir = 2
                    if door_direction_absolute == 2:
                        door_dir = 1
                    if door_direction_absolute == 3:
                        door_dir = 3
                    if door_direction_absolute == 4:
                        door_dir = 0
                if current_direction == 2: # east
                    if door_direction_absolute == 1:
                        door_dir = 0
                    if door_direction_absolute == 2:
                        door_dir = 2
                    if door_direction_absolute == 3:
                        door_dir = 1
                    if door_direction_absolute == 4:
                        door_dir = 3
                if current_direction == 3: # south
                    if door_direction_absolute == 1:
                        door_dir = 3
                    if door_direction_absolute == 2:
                        door_dir = 0
                    if door_direction_absolute == 3:
                        door_dir = 2
                    if door_direction_absolute == 4:
                        door_dir = 1
                if current_direction == 4: # west
                    if door_direction_absolute == 1:
                        door_dir = 1
                    if door_direction_absolute == 2:
                        door_dir = 3
                    if door_direction_absolute == 3:
                        door_dir = 0
                    if door_direction_absolute == 4:
                        door_dir = 2


                if door_dir == 0:
                    left_doors.append(paired_door)
                elif door_dir == 1:
                    right_doors.append(paired_door)
                else:
                    other_doors.append(paired_door)

                if next_direction != current_direction:
                    if current_direction == 1:
                        if next_direction == 4:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 2:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    if current_direction == 2:
                        if next_direction == 1:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 3:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    if current_direction == 3:
                        if next_direction == 2:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 4:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                    if current_direction == 4:
                        if next_direction == 3:
                            advice.append("Proceed to "+ count_translation[len(left_doors)] + " door on your left. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])
                        if next_direction == 1:
                            advice.append("Proceed to "+ count_translation[len(right_doors)] + " door on your right. ")
                            need_to_reach_destinations.append(decision_points[path[index]]['location'])

                    right_doors = []
                    left_doors = []

        if decision_points[path[index]]['type'] == 'door':
            if index > 0 and decision_points[path[index]]['type'] == 'entrance':
                advice.append('Proceed through door. ')
                need_to_reach_destinations.append(decision_points[path[index]]['location'])

        if decision_points[path[index]]['type'] == 'intersection':
            if index > 0:
                intersection_count += 1
                if next_direction != current_direction:
                    # if intersection_count > 0:
                    #     print("intersection count", intersection_count)
                    #     print('prev_direction', prev_direction)
                    #     print('current_direction', current_direction)
                    #     print("next_direction", next_direction)
                    #     print()

                    advice.append('Proceed to ' + count_translation[intersection_count] + ' intersection.')
                    need_to_reach_destinations.append(decision_points[path[index]]['location'])

                    if current_direction == 1:
                        if next_direction == 4:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 2:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                    if current_direction == 2:
                        if next_direction == 1:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 3:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                    if current_direction == 3:
                        if next_direction == 2:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 4:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                    if current_direction == 4:
                        if next_direction == 3:
                            advice.append("Turn left. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        elif next_direction == 1:
                            advice.append("Turn right. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])
                        else:
                            advice.append("Turn around. ")
                            need_to_reach_destinations.append(
                                decision_points[path[index]]['location'])

                    intersection_count = 0
                    right_doors = []
                    left_doors = []

        prev_direction = current_direction
        current_direction = next_direction

    if decision_points[path[-1]]['type'] == 'victim':
        advice.append('Find and triage ' + color_dict[decision_points[path[-1]]['color']] + ' victim.')
        need_to_reach_destinations.append(decision_points[path[-1]]['location'])

    generated_advice = ''

    for ad in advice[:3]:
        generated_advice += ad

    if len(advice) > 2:
        final_destination = need_to_reach_destinations[2]
    else:
        final_destination = decision_points[len(path)-1]['location']

    # for ad in advice:
    #     generated_advice += ad
    #
    # final_destination = decision_points[len(path)-1]['location']
    # print("whole path ", [path[c] for c in range(len(path))])
    # print('advice', advice)
    return generated_advice, final_destination



def generate_level_1_astar_decision_points_w_rotation(prev_loc, gameboard, obstacles, yellow_locs, green_locs, doors, stairs,
                            goal_tuple_to_id, id_to_goal_tuple, reversed_complete_room_dict,
                            player, start_idx, goal_idx, decision_points, decision_neighbors, victim_idx, curr_coordinate, prev_coordinate):
    decision_search = Decision_Point_Search(decision_neighbors, decision_points)
    decision_path = decision_search.a_star(start_idx, goal_idx)

    if len(decision_path) >=2:
        target_location = decision_path[2]
    else:
        target_location = decision_path[-1]
    path = recompute_path((player.x, player.y), decision_points[target_location]['location'], id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs)

    start_direction = player.current_direction  # 1234 = NESW
    loc_tuples = path
    x_waypoints = [loc[0] for loc in loc_tuples]
    z_waypoints = [loc[1] for loc in loc_tuples]

    current_advice = ('Walk forward ', 0)
    advice = ''
    advice_list = []
    need_to_reach_destinations = []

    direction_dict = {
        1: 'N',
        2: 'E',
        3: 'S',
        4: 'W',
    }
    color_dict = {
        'y': 'yellow',
        'g': 'green'
    }
    time_dict = {
        'y': '15',
        'g': '10'
    }

    if len(path) <= 1:
        return 'Approach and save ' + color_dict[decision_points[decision_path[-1]]['color']] + ' victim by pressing SPACE '+time_dict[decision_points[decision_path[-1]]['color']]+' times. ', decision_points[target_location]['location']


    # Check if player is past the current start location
    if manhattan_dist(player.x, player.y, decision_points[decision_path[1]]['location'][0],
                      decision_points[decision_path[1]]['location'][1]) < \
            manhattan_dist(decision_points[decision_path[0]]['location'][0], decision_points[decision_path[1]]['location'][1],
                           decision_points[decision_path[1]]['location'][0], decision_points[decision_path[1]]['location'][1]):
        # player is closer to second goal point than first goal point
        start_decision_point_idx = 1
    else:
        # player is closer to first goal point
        start_decision_point_idx = 0

    if start_decision_point_idx == 0:
        current_direction, (dx, dy) = check_approximate_direction(player.x,
                                                                  player.y,
                                                                  decision_points[decision_path[start_decision_point_idx]][
                                                                      'location'][0],
                                                                  decision_points[decision_path[start_decision_point_idx]][
                                                                      'location'][1])
        if dx + dy < 2:
            current_direction, (dx, dy) = check_approximate_direction(
                decision_points[decision_path[start_decision_point_idx]]['location'][0],
                decision_points[decision_path[start_decision_point_idx]]['location'][1],
                decision_points[decision_path[start_decision_point_idx + 1]]['location'][0],
                decision_points[decision_path[start_decision_point_idx + 1]]['location'][1])

    else:
        current_direction, (dx, dy) = check_approximate_direction(
            decision_points[decision_path[start_decision_point_idx - 1]]['location'][0],
            decision_points[decision_path[start_decision_point_idx - 1]]['location'][1],
            decision_points[decision_path[start_decision_point_idx]]['location'][0],
            decision_points[decision_path[start_decision_point_idx]]['location'][1])

    if current_direction != start_direction:
        if start_direction == 1:
            if current_direction == 4:
                advice_list.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            elif current_direction == 2:
                advice_list.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            else:
                advice_list.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
        if start_direction == 2:
            if current_direction == 1:
                advice_list.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            elif current_direction == 3:
                advice_list.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            else:
                advice_list.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
        if start_direction == 3:
            if current_direction == 2:
                advice_list.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            elif current_direction == 4:
                advice_list.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            else:
                advice_list.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
        if start_direction == 4:
            if current_direction == 3:
                advice_list.append("Turn left. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            elif current_direction == 1:
                advice_list.append("Turn right. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])
            else:
                advice_list.append("Turn around. ")
                need_to_reach_destinations.append(decision_points[decision_path[start_decision_point_idx]]['location'])




    for j in range(1, len(x_waypoints)-1):
        prev_loc = (x_waypoints[j-1], z_waypoints[j-1])
        curr_loc = (x_waypoints[j], z_waypoints[j])
        next_loc = (x_waypoints[j + 1], z_waypoints[j + 1])

        curr_direction = check_approximate_direction(prev_loc[0], prev_loc[1], curr_loc[0], curr_loc[1])
        next_direction = check_approximate_direction(curr_loc[0], curr_loc[1], next_loc[0], next_loc[1])

        if next_direction == curr_direction:
            if next_direction == NORTH or next_direction == SOUTH:
                current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            else:
                current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))
        else:
            advice_list.append(get_turn_direction(curr_direction, next_direction))
            need_to_reach_destinations.append((x_waypoints[j], z_waypoints[j]))

            if current_advice[1] > 0:
                # advice += "Walk " + str(current_advice[1]) + ' steps '+curr_direction+'. '
                # advice += "Face " + next_direction + '. '

                # advice_list.append("Face " + next_direction + '. ')

                advice_list.append("Walk " + str(current_advice[1]) + ' steps'+'. ')
                need_to_reach_destinations.append((x_waypoints[j], z_waypoints[j]))

            current_advice = ('Walk forward ', 0)
            # if next_direction == 'N' or next_direction == 'S':
            #     current_advice = ('Walk forward ', current_advice[1] + abs(z_waypoints[j] - prev_loc[1]))
            # else:
            #     current_advice = ('Walk forward ', current_advice[1] + abs(x_waypoints[j] - prev_loc[0]))

    if current_advice[1] > 0:
        # advice += "Walk " + str(current_advice[1]) + ' steps ' + curr_direction + '. '
        turns = get_turn_direction(curr_direction, next_direction)
        if turns != '':
            advice_list.append(turns)
        advice_list.append("Walk " + str(current_advice[1]) + ' steps' + '. ')
        # advice_list.append("Walk " + str(current_advice[1]) + ' steps ' + curr_direction + '. ')
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))

    if (x_waypoints[-1], z_waypoints[-1]) in yellow_locs:
        advice += 'Press SPACE 15x to save victim. '
        advice_list.append('Press SPACE 15x to save victim. ')
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))

    if (x_waypoints[-1], z_waypoints[-1]) in green_locs:
        advice += 'Press SPACE 10x to save victim. '
        advice_list.append('Press SPACE 10x to save victim. ')
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))
        need_to_reach_destinations.append((x_waypoints[-1], z_waypoints[-1]))

    # advice_list = [x+'. ' for x in advice.split('. ')]

    if len(advice_list) > 3:
        final_destination = need_to_reach_destinations[2]
        advice_list = advice_list[:3]
    else:
        final_destination = (x_waypoints[-1], z_waypoints[-1])
    final_advice = ' '.join(advice_list)

    return final_advice, final_destination

def get_turn_direction(start_direction, current_direction):
    advice = []
    if current_direction != start_direction:
        if start_direction == 1:
            if current_direction == 4:
                advice.append("Turn left. ")
            elif current_direction == 2:
                advice.append("Turn right. ")
            else:
                advice.append("Turn around. ")
        if start_direction == 2:
            if current_direction == 1:
                advice.append("Turn left. ")
            elif current_direction == 3:
                advice.append("Turn right. ")
            else:
                advice.append("Turn around. ")
        if start_direction == 3:
            if current_direction == 2:
                advice.append("Turn left. ")
            elif current_direction == 4:
                advice.append("Turn right. ")
            else:
                advice.append("Turn around. ")
        if start_direction == 4:
            if current_direction == 3:
                advice.append("Turn left. ")
            elif current_direction == 1:
                advice.append("Turn right. ")
            else:
                advice.append("Turn around. ")
    if len(advice) >= 1:
        return advice[0]
    return ''