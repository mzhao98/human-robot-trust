import pandas as pd
import numpy as np
import json
import signal
import matplotlib.pyplot as plt
import copy
import heapq
import pickle as pkl
import sys, time, random
import matplotlib.ticker as ticker
prefix = './mission/static/js/py_files/'
from mission.static.js.py_files.a_star import Search, Decision_Point_Search, Decision_Point_Search_Augmented
from mission.static.js.py_files.gameboard_utils import *
from mission.static.js.py_files.settings import *
from mission.static.js.py_files.advice_utils import recompute_path

import signal, time

class Timeout():
  """Timeout class using ALARM signal"""
  class Timeout(Exception): pass

  def __init__(self, sec):
    self.sec = sec

  def __enter__(self):
    signal.signal(signal.SIGALRM, self.raise_timeout)
    signal.alarm(self.sec)

  def __exit__(self, *args):
    signal.alarm(0) # disable alarm

  def raise_timeout(self, *args):
    raise Timeout.Timeout()


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

def euclidean(tup1, tup2):
    return np.sqrt((tup1[0]-tup2[0])**2 + (tup1[1]-tup2[1])**2)


def recompute_path(current_location, target_location, id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs):
    # print("current_location, target_location", (current_location, target_location))
    # print("gameboard", gameboard.shape)
    single_search = Search(gameboard, current_location, target_location, obstacles, yellow_locs, green_locs)
    travel_path = single_search.a_star_new(current_location, target_location)
    return travel_path

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


class Level_2_Instruction:
    def __init__(self):
        self.turn_advice_dict = {
            LEFT: "left",
            RIGHT: "right",
            AROUND: "around",
            AHEAD: 'ahead',
            BEHIND: "behind",
        }

        self.room_side_turn_advice_dict = {
            LEFT: "on your left",
            RIGHT: "on your right",
            AROUND: "behind you",
            AHEAD: 'up ahead',
            BEHIND: "behind you",
        }

        self.aug_turn_advice_dict = {
            LEFT: "on your left",
            RIGHT: "on your right",
            AROUND: "behind you",
            AHEAD: 'up ahead',
            BEHIND: "behind you",
        }

        self.nth_advice_dict = {
            0: "first",
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
        }
        self.compass_advice_dict = {
            NORTH: "North",
            SOUTH: "South",
            EAST: "East",
            WEST: "West",
        }
        self.complexity_level = 2

        self.inverse_gameboard, self.obstacles, self.yellow_locs, self.green_locs, self.doors, self.stairs, self.obstacles_stairs_free = get_inv_gameboard()
        self.goal_tuple_to_id, self.goal_tuple_to_id_tuplekeys, self.id_to_goal_tuple, self.distance_dict, \
        self.distance_matrix = load_goal_distances(
            cutoff_victims=34)

        self.rooms_dictionary, self.position_to_room_dict, self.complete_room_dict, self.reversed_complete_room_dict = get_rooms_dict(
            self.inverse_gameboard)

        self.decision_points, self.dp_intersections, self.dp_entrances, self.dp_doors = self.decision_point_dict()

        self.saved_victims_indices = []
        self.expired_yellow_indices = []

        simple_map = prefix + 'setup_files/map.csv'
        simple_map_df = pd.read_csv(simple_map)

        with open(prefix + 'setup_files/aggregate_path_22vic_2.pkl', 'rb') as file:
            self.victim_path = pkl.load(file)

        self.gameboard = np.empty((50, 93, 3))


        # Create gameboard dictionary
        self.gameboard_dict = {}
        for x_coor in range(self.gameboard.shape[0]):
            for z_coor in range(self.gameboard.shape[1]):
                self.gameboard_dict[(x_coor, z_coor)] = {}
                self.gameboard_dict[(x_coor, z_coor)]['open'] = 1
                self.gameboard_dict[(x_coor, z_coor)]['type'] = 'tile'

        self.gameboard[:, :, 0].fill(255)
        self.gameboard[:, :, 1].fill(255)
        self.gameboard[:, :, 2].fill(255)

        for (x_coor, z_coor) in [(8, 30), (8, 31), (8, 32), (8, 33)]:
            self.gameboard[x_coor, z_coor] = [185, 116, 85]
            self.gameboard_dict[(x_coor, z_coor)]['open'] = 1
            self.gameboard_dict[(x_coor, z_coor)]['door'] = 'door'
            self.doors.append((z_coor, x_coor))

        for index, row in simple_map_df.iterrows():
            x_coor = row['x']
            z_coor = row['z']

            if row['key'] == 'walls':
                self.gameboard[z_coor, x_coor] = [237, 184, 121]
                self.gameboard_dict[(z_coor, x_coor)]['open'] = 0
                self.gameboard_dict[(z_coor, x_coor)]['type'] = 'wall'
            elif row['key'] == 'doors':
                self.gameboard[z_coor, x_coor] = [185, 116, 85]
                self.gameboard_dict[(z_coor, x_coor)]['open'] = 1
                self.gameboard_dict[(z_coor, x_coor)]['door'] = 'door'
            elif row['key'] == 'stairs':
                self.gameboard[z_coor, x_coor] = [153, 76, 0]
                self.gameboard_dict[(z_coor, x_coor)]['open'] = 1
                self.gameboard_dict[(z_coor, x_coor)]['type'] = 'stair'

        # DROP yellow locs and green locs not in id_to_goal_tuple
        yellow_drop = []
        for i in range(len(self.yellow_locs)):
            if (self.yellow_locs[i][1], self.yellow_locs[i][0]) not in self.id_to_goal_tuple.values():
                yellow_drop.append(i)
        self.yellow_locs = [i for j, i in enumerate(self.yellow_locs) if j not in yellow_drop]

        green_drop = []
        for i in range(len(self.green_locs)):
            if (self.green_locs[i][1], self.green_locs[i][0]) not in self.id_to_goal_tuple.values():
                green_drop.append(i)
        self.green_locs = [i for j, i in enumerate(self.green_locs) if j not in green_drop]

        # Flip self.yellow_locs, self.green_locs, and self.complete_room_dict so that they are (x,z)
        self.flipped_yellow_locs = self.yellow_locs
        self.flipped_green_locs = self.green_locs
        self.flipped_complete_room_dict = self.complete_room_dict

        self.yellow_locs = [(j, i) for (i, j) in self.yellow_locs]
        self.green_locs = [(j, i) for (i, j) in self.green_locs]

        for room_name in self.complete_room_dict:
            self.complete_room_dict[room_name] = [(j, i) for (i, j) in self.complete_room_dict[room_name]]

        # Flip self.obstacles, self.stairs, and self.doors so that they are (x,z)
        self.flipped_obstacles = self.obstacles
        self.flipped_obstacles_stairs_free = self.obstacles_stairs_free
        self.flipped_stairs = self.stairs
        self.flipped_doors = self.doors

        self.obstacles = [(j, i) for (i, j) in self.obstacles]
        self.stairs = [(j, i) for (i, j) in self.stairs]
        self.doors = [(j, i) for (i, j) in self.doors]

        with open(prefix + "setup_files/intersection_floodfill_dict.pkl", 'rb') as filename:
            self.intersection_floodfill_dict = pkl.load(filename)
        with open(prefix + "setup_files/door_floodfill_dict.pkl", 'rb') as filename:
            self.door_floodfill_dict = pkl.load(filename)

        with open(prefix + "setup_files/intersection_locations_dict.pkl", 'rb') as filename:
            self.intersection_locations_dict = pkl.load(filename)
        with open(prefix + "setup_files/entrance_locations_dict.pkl", 'rb') as filename:
            self.entrance_locations_dict = pkl.load(filename)

        self.make_doors_list()

        self.entrance_location_to_door = {}
        for door_idx in self.doors_details_dict:
            for entrance_loc in self.doors_details_dict[door_idx]['entrance_locations']:
                self.entrance_location_to_door[entrance_loc] = {"door_index": door_idx,
                                                                'door_location': self.doors_details_dict[door_idx][
                                                                    'location']}

        with open(prefix + 'setup_files/shortest_augmented_dps_distances_2.pkl', 'rb') as handle:
            self.closest_decision_pt = pkl.load(handle)

        with open(prefix + "setup_files/door_index_to_dp_indices_dict.pkl", 'rb') as handle:
            self.door_index_to_dp_indices_dict = pkl.load(handle)

        with open(prefix + 'setup_files/floodfill_dict.pkl', 'rb') as filename:
            self.gameboard_floodfill_dictionary = pkl.load(filename)

        self.previous_likelihood_dict = {}
        for candidate_idx in self.id_to_goal_tuple:
            self.previous_likelihood_dict[candidate_idx] = [1/4] * 20

        self.prev_location_on_traj = (6, 5)
        self.prev_dp_index = 0
        self.prev_dp_list = []
        self.previous_advice = ""
        self.previous_dp_destination = 0
        self.previous_keep_dp_destination = 0
        self.level_change_counter = 0
        self.level = 2
        self.previously_saved = []
        self.visited_rooms_list = []

        self.prev_past_traj = []

        self.level_2_hold_counter = 5
        self.level_2_hold_advice = ''
        self.level_2_static_hold_counter = 5
        self.level_2_static_hold_advice = ''

        self.generate_victim_path_details()
        self.dp_dict = self.generate_decision_points_augmented()

    def generate_decision_points_augmented(self):
        all_decision_points_xy = {
            0: {
                "type": "start",
                "location": (6, 5),
                "neighbors": [1, 2, 3, 4],
                "within_room_name": "Start",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            1: {
                "type": "entrance_out",
                "location": (15, 1),
                "neighbors": [0, 5],
                "within_room_name": "Start",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            2: {
                "type": "entrance_out",
                "location": (15, 2),
                "neighbors": [0, 6],
                "within_room_name": "Start",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            3: {
                "type": "entrance_out",
                "location": (15, 4),
                "neighbors": [0, 7],
                "within_room_name": "Start",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            4: {
                "type": "entrance_out",
                "location": (15, 5),
                "neighbors": [0, 8],
                "within_room_name": "Start",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            5: {
                "type": "door",
                "location": (16, 1),
                "neighbors": [1, 9],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            6: {
                "type": "door",
                "location": (16, 2),
                "neighbors": [2, 10],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            7: {
                "type": "door",
                "location": (16, 4),
                "neighbors": [3, 11],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            8: {
                "type": "door",
                "location": (16, 5),
                "neighbors": [4, 12],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [(15, 1), (15, 2), (15, 4), (15, 5)],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [1, 2, 3, 4],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            9: {
                "type": "entrance_in",
                "location": (17, 1),
                "neighbors": [5, 13],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            10: {
                "type": "entrance_in",
                "location": (17, 2),
                "neighbors": [6, 13],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            11: {
                "type": "entrance_in",
                "location": (17, 4),
                "neighbors": [7, 13],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            12: {
                "type": "entrance_in",
                "location": (17, 5),
                "neighbors": [8, 13],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [(16, 1), (16, 2), (16, 4), (16, 5)],
                "door_dp_indices_list": [5, 6, 7, 8],
                "entrance_in_coordinates_list": [(17, 1), (17, 2), (17, 4), (17, 5)],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [9, 10, 11, 12],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            13: {
                "type": "intersection",
                "location": (20, 4),
                "neighbors": [9, 10, 11, 12, 14, 15, 16],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Start",
                "entering_room_name": "Starting Deck",
            },
            14: {
                "type": "entrance_in",
                "location": (18, 6),
                "neighbors": [13, 17],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            15: {
                "type": "entrance_in",
                "location": (20, 6),
                "neighbors": [13, 18],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            16: {
                "type": "entrance_in",
                "location": (22, 6),
                "neighbors": [13, 19],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            17: {
                "type": "door",
                "location": (18, 7),
                "neighbors": [14, 20],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            18: {
                "type": "door",
                "location": (20, 7),
                "neighbors": [15, 21],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            19: {
                "type": "door",
                "location": (22, 7),
                "neighbors": [16, 22],
                "within_room_name": "Starting Deck",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            20: {
                "type": "entrance_out",
                "location": (18, 8),
                "neighbors": [17, 23],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            21: {
                "type": "entrance_out",
                "location": (20, 8),
                "neighbors": [18, 23],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            22: {
                "type": "entrance_out",
                "location": (22, 8),
                "neighbors": [19, 23],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [(18, 7), (20, 7), (22, 7)],
                "door_dp_indices_list": [17, 18, 19],
                "entrance_in_coordinates_list": [(18, 6), (20, 6), (22, 6)],
                "entrance_out_coordinates_list": [(18, 8), (20, 8), (22, 8)],
                "entrance_in_dp_indices_list": [14, 15, 16],
                "entrance_out_dp_indices_list": [20, 21, 22],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Starting Deck",
                "entering_room_name": "North Hallway",
            },
            23: {
                "type": "intersection",
                "location": (20, 10),
                "neighbors": [20, 21, 22, 24],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            24: {
                "type": "passing_door",
                "location": (25, 10),
                "neighbors": [23, 25, 28, 31],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [3, 21],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            25: {
                "type": "entrance_out",
                "location": (25, 9),
                "neighbors": [26, 24],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 3,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Room 100",
            },
            26: {
                "type": "door",
                "location": (25, 8),
                "neighbors": [27, 25],
                "within_room_name": "Room 100",
                "color": "",
                "door_index": 3,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Room 100",
            },
            27: {
                "type": "entrance_in",
                "location": (25, 7),
                "neighbors": [26],
                "within_room_name": "Room 100",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Room 100",
            },
            28: {
                "type": "entrance_out",
                "location": (25, 12),
                "neighbors": [24, 29],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 21,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "The Computer Farm",
            },
            29: {
                "type": "door",
                "location": (25, 13),
                "neighbors": [28, 30],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": 21,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "The Computer Farm",
            },
            30: {
                "type": "entrance_in",
                "location": (25, 14),
                "neighbors": [29, 162, 161, 165],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": 21,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "The Computer Farm",
            },
            31: {
                "type": "passing_door",
                "location": (31, 10),
                "neighbors": [24, 32, 33, 34, 35, 46],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [4],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            32: {
                "type": "entrance_out",
                "location": (30, 9),
                "neighbors": [31, 36],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            33: {
                "type": "entrance_out",
                "location": (31, 9),
                "neighbors": [31, 37],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            34: {
                "type": "entrance_out",
                "location": (32, 9),
                "neighbors": [31, 38],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            35: {
                "type": "entrance_out",
                "location": (33, 9),
                "neighbors": [31, 39],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            36: {
                "type": "door",
                "location": (30, 8),
                "neighbors": [32, 40],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            37: {
                "type": "door",
                "location": (31, 8),
                "neighbors": [33, 41],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            38: {
                "type": "door",
                "location": (32, 8),
                "neighbors": [34, 42],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            39: {
                "type": "door",
                "location": (33, 8),
                "neighbors": [35, 43],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            40: {
                "type": "entrance_in",
                "location": (30, 7),
                "neighbors": [36, 44, 45],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            41: {
                "type": "entrance_in",
                "location": (31, 7),
                "neighbors": [37, 44, 45],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            42: {
                "type": "entrance_in",
                "location": (32, 7),
                "neighbors": [38, 44, 45],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            43: {
                "type": "entrance_in",
                "location": (33, 7),
                "neighbors": [39, 44, 45],
                "within_room_name": "Open Break Area",
                "color": "",
                "door_index": 4,
                "door_coordinates_list": [(30, 8), (31, 8), (32, 8), (33, 8)],
                "door_dp_indices_list": [36, 37, 38, 39],
                "entrance_in_coordinates_list": [(30, 7), (31, 7), (32, 7), (33, 7)],
                "entrance_out_coordinates_list": [(30, 9), (31, 9), (32, 9), (33, 9)],
                "entrance_in_dp_indices_list": [40, 41, 42, 43],
                "entrance_out_dp_indices_list": [32, 33, 34, 35],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Open Break Area",
            },
            44: {
                "type": "victim",
                "location": (30, 4),
                "neighbors": [40, 41, 42, 43, 45],
                "within_room_name": "Open Break Area",
                "color": "green",
                "door_index": 4,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            45: {
                "type": "victim",
                "location": (35, 7),
                "neighbors": [40, 41, 42, 43, 44],
                "within_room_name": "Open Break Area",
                "color": "yellow",
                "door_index": 4,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            46: {
                "type": "intersection",
                "location": (36, 10),
                "neighbors": [31, 47, 147],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            47: {
                "type": "passing_door",
                "location": (37, 10),
                "neighbors": [46, 48, 54],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 5,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [5],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            48: {
                "type": "entrance_out",
                "location": (37, 9),
                "neighbors": [47, 49],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 5,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Executive Suite 1",
            },
            49: {
                "type": "door",
                "location": (37, 8),
                "neighbors": [48, 50],
                "within_room_name": "Executive Suite 1",
                "color": "",
                "door_index": 5,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Executive Suite 1",
            },
            50: {
                "type": "entrance_in",
                "location": (37, 7),
                "neighbors": [49, 51],
                "within_room_name": "Executive Suite 1",
                "color": "",
                "door_index": 5,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Executive Suite 1",
            },
            51: {
                "type": "room_corner",
                "location": (37, 2),
                "neighbors": [50, 52, 53],
                "within_room_name": "Executive Suite 1",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            52: {
                "type": "victim",
                "location": (48, 6),
                "neighbors": [51, 53],
                "within_room_name": "Executive Suite 1",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            53: {
                "type": "victim",
                "location": (50, 5),
                "neighbors": [52, 51],
                "within_room_name": "Executive Suite 1",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            54: {
                "type": "passing_door",
                "location": (41, 10),
                "neighbors": [47, 160, 55],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 22,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [22],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            55: {
                "type": "passing_door",
                "location": (54, 10),
                "neighbors": [54, 56, 60],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 6,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Executive Suite 2",
            },
            56: {
                "type": "entrance_out",
                "location": (54, 9),
                "neighbors": [55, 57],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 6,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Executive Suite 2",
            },
            57: {
                "type": "door",
                "location": (54, 8),
                "neighbors": [56, 58],
                "within_room_name": "Executive Suite 2",
                "color": "",
                "door_index": 6,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Executive Suite 2",
            },
            58: {
                "type": "entrance_in",
                "location": (54, 7),
                "neighbors": [57, 59],
                "within_room_name": "Executive Suite 2",
                "color": "",
                "door_index": 6,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            59: {
                "type": "room_corner",
                "location": (54, 2),
                "neighbors": [58],
                "within_room_name": "Executive Suite 2",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            60: {
                "type": "intersection",
                "location": (58, 10),
                "neighbors": [55, 61, 121],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            61: {
                "type": "passing_door",
                "location": (69, 10),
                "neighbors": [60, 62, 63, 74],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [(69, 8), (70, 8)],
                "door_dp_indices_list": [64, 65],
                "entrance_in_coordinates_list": [(69, 7), (70, 7)],
                "entrance_out_coordinates_list": [(69, 9), (70, 9)],
                "entrance_in_dp_indices_list": [66, 67],
                "entrance_out_dp_indices_list": [62, 63],
                "passing_associated_door_indices": [7],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "King Chris\' Office",
            },
            62: {
                "type": "entrance_out",
                "location": (69, 9),
                "neighbors": [61, 64],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [(69, 8), (70, 8)],
                "door_dp_indices_list": [64, 65],
                "entrance_in_coordinates_list": [(69, 7), (70, 7)],
                "entrance_out_coordinates_list": [(69, 9), (70, 9)],
                "entrance_in_dp_indices_list": [66, 67],
                "entrance_out_dp_indices_list": [62, 63],
                "passing_associated_door_indices": [7],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "King Chris\' Office",
            },
            63: {
                "type": "entrance_out",
                "location": (70, 9),
                "neighbors": [61, 65],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [(69, 8), (70, 8)],
                "door_dp_indices_list": [64, 65],
                "entrance_in_coordinates_list": [(69, 7), (70, 7)],
                "entrance_out_coordinates_list": [(69, 9), (70, 9)],
                "entrance_in_dp_indices_list": [66, 67],
                "entrance_out_dp_indices_list": [62, 63],
                "passing_associated_door_indices": [7],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "King Chris\' Office",
            },
            64: {
                "type": "door",
                "location": (69, 8),
                "neighbors": [62, 66],
                "within_room_name": "King Chris\' Office",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [(69, 8), (70, 8)],
                "door_dp_indices_list": [64, 65],
                "entrance_in_coordinates_list": [(69, 7), (70, 7)],
                "entrance_out_coordinates_list": [(69, 9), (70, 9)],
                "entrance_in_dp_indices_list": [66, 67],
                "entrance_out_dp_indices_list": [62, 63],
                "passing_associated_door_indices": [7],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "King Chris\' Office",
            },
            65: {
                "type": "door",
                "location": (70, 8),
                "neighbors": [63, 67],
                "within_room_name": "King Chris\' Office",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [(69, 8), (70, 8)],
                "door_dp_indices_list": [64, 65],
                "entrance_in_coordinates_list": [(69, 7), (70, 7)],
                "entrance_out_coordinates_list": [(69, 9), (70, 9)],
                "entrance_in_dp_indices_list": [66, 67],
                "entrance_out_dp_indices_list": [62, 63],
                "passing_associated_door_indices": [7],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "King Chris\' Office",
            },
            66: {
                "type": "entrance_in",
                "location": (69, 7),
                "neighbors": [64, 68],
                "within_room_name": "King Chris\' Office",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [(69, 8), (70, 8)],
                "door_dp_indices_list": [62, 63],
                "entrance_in_coordinates_list": [(69, 7), (70, 7)],
                "entrance_out_coordinates_list": [(69, 9), (70, 9)],
                "entrance_in_dp_indices_list": [66, 67],
                "entrance_out_dp_indices_list": [62, 63],
                "passing_associated_door_indices": [7],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "King Chris\' Office",
            },
            67: {
                "type": "entrance_in",
                "location": (70, 7),
                "neighbors": [65, 68],
                "within_room_name": "King Chris\' Office",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [(69, 8), (70, 8)],
                "door_dp_indices_list": [62, 63],
                "entrance_in_coordinates_list": [(69, 7), (70, 7)],
                "entrance_out_coordinates_list": [(69, 9), (70, 9)],
                "entrance_in_dp_indices_list": [66, 67],
                "entrance_out_dp_indices_list": [62, 63],
                "passing_associated_door_indices": [7],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "King Chris\' Office",
            },
            68: {
                "type": "room_corner",
                "location": (69, 2),
                "neighbors": [66, 67, 69, 70],
                "within_room_name": "King Chris\' Office",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            69: {
                "type": "victim",
                "location": (74, 6),
                "neighbors": [68, 70],
                "within_room_name": "King Chris\' Office",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            70: {
                "type": "entrance_out",
                "location": (83, 8),
                "neighbors": [69, 68, 71],
                "within_room_name": "King Chris\' Office",
                "color": "",
                "door_index": 8,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "King Chris\' Office",
                "entering_room_name": "Kings Terrace",
            },
            71: {
                "type": "door",
                "location": (84, 8),
                "neighbors": [70, 72],
                "within_room_name": "Kings Terrace",
                "color": "",
                "door_index": 8,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "King Chris\' Office",
                "entering_room_name": "Kings Terrace",
            },
            72: {
                "type": "entrance_in",
                "location": (86, 8),
                "neighbors": [71, 73],
                "within_room_name": "Kings Terrace",
                "color": "",
                "door_index": 8,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "King Chris\' Office",
                "entering_room_name": "Kings Terrace",
            },
            73: {
                "type": "victim",
                "location": (87, 4),
                "neighbors": [72],
                "within_room_name": "Kings Terrace",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            74: {
                "type": "intersection",
                "location": (72, 10),
                "neighbors": [61, 75],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            75: {
                "type": "passing_door",
                "location": (72, 15),
                "neighbors": [76, 81, 87],
                "within_room_name": "Rear Bridge",
                "color": "",
                "door_index": 32,
                "door_coordinates_list": [(70, 15), (70, 16)],
                "door_dp_indices_list": [77, 80],
                "entrance_in_coordinates_list": [(69, 15), (69, 16)],
                "entrance_out_coordinates_list": [(71, 15), (71, 16)],
                "entrance_in_dp_indices_list": [78, 79],
                "entrance_out_dp_indices_list": [76, 81],
                "passing_associated_door_indices": [32],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            76: {
                "type": "entrance_out",
                "location": (71, 15),
                "neighbors": [77, 75],
                "within_room_name": "Rear Bridge",
                "color": "",
                "door_index": 32,
                "door_coordinates_list": [(70, 15), (70, 16)],
                "door_dp_indices_list": [77, 80],
                "entrance_in_coordinates_list": [(69, 15), (69, 16)],
                "entrance_out_coordinates_list": [(71, 15), (71, 16)],
                "entrance_in_dp_indices_list": [78, 79],
                "entrance_out_dp_indices_list": [76, 81],
                "passing_associated_door_indices": [32],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            77: {
                "type": "door",
                "location": (70, 15),
                "neighbors": [78, 76],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 32,
                "door_coordinates_list": [(70, 15), (70, 16)],
                "door_dp_indices_list": [77, 80],
                "entrance_in_coordinates_list": [(69, 15), (69, 16)],
                "entrance_out_coordinates_list": [(71, 15), (71, 16)],
                "entrance_in_dp_indices_list": [78, 79],
                "entrance_out_dp_indices_list": [76, 81],
                "passing_associated_door_indices": [32],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            78: {
                "type": "entrance_in",
                "location": (69, 15),
                "neighbors": [77, 83, 82],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 32,
                "door_coordinates_list": [(70, 15), (70, 16)],
                "door_dp_indices_list": [77, 80],
                "entrance_in_coordinates_list": [(69, 15), (69, 16)],
                "entrance_out_coordinates_list": [(71, 15), (71, 16)],
                "entrance_in_dp_indices_list": [78, 79],
                "entrance_out_dp_indices_list": [76, 81],
                "passing_associated_door_indices": [32],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            79: {
                "type": "entrance_in",
                "location": (69, 16),
                "neighbors": [80, 83, 82],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 32,
                "door_coordinates_list": [(70, 15), (70, 16)],
                "door_dp_indices_list": [77, 80],
                "entrance_in_coordinates_list": [(69, 15), (69, 16)],
                "entrance_out_coordinates_list": [(71, 15), (71, 16)],
                "entrance_in_dp_indices_list": [78, 79],
                "entrance_out_dp_indices_list": [76, 81],
                "passing_associated_door_indices": [32],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            80: {
                "type": "door",
                "location": (70, 16),
                "neighbors": [79, 81],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 32,
                "door_coordinates_list": [(70, 15), (70, 16)],
                "door_dp_indices_list": [77, 80],
                "entrance_in_coordinates_list": [(69, 15), (69, 16)],
                "entrance_out_coordinates_list": [(71, 15), (71, 16)],
                "entrance_in_dp_indices_list": [78, 79],
                "entrance_out_dp_indices_list": [76, 81],
                "passing_associated_door_indices": [32],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            81: {
                "type": "entrance_out",
                "location": (71, 16),
                "neighbors": [80, 75],
                "within_room_name": "Rear Bridge",
                "color": "",
                "door_index": 32,
                "door_coordinates_list": [(70, 15), (70, 16)],
                "door_dp_indices_list": [77, 80],
                "entrance_in_coordinates_list": [(69, 15), (69, 16)],
                "entrance_out_coordinates_list": [(71, 15), (71, 16)],
                "entrance_in_dp_indices_list": [78, 79],
                "entrance_out_dp_indices_list": [76, 81],
                "passing_associated_door_indices": [32],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            82: {
                "type": "victim",
                "location": (67, 18),
                "neighbors": [78, 79, 83],
                "within_room_name": "Herbalife Conference Room",
                "color": "yellow",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            83: {
                "type": "room_center",
                "location": (65, 24),
                "neighbors": [82, 84, 78, 79, 85, 86],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            84: {
                "type": "victim",
                "location": (65, 30),
                "neighbors": [83, 85, 86],
                "within_room_name": "Herbalife Conference Room",
                "color": "yellow",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            85: {
                "type": "entrance_in",
                "location": (62, 32),
                "neighbors": [84, 83, 119],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 31,
                "door_coordinates_list": [(61, 32), (62, 33)],
                "door_dp_indices_list": [119, 118],
                "entrance_in_coordinates_list": [(62, 32), (62, 33)],
                "entrance_out_coordinates_list": [(60, 32), (60, 33)],
                "entrance_in_dp_indices_list": [85, 86],
                "entrance_out_dp_indices_list": [116, 117],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            86: {
                "type": "entrance_in",
                "location": (62, 33),
                "neighbors": [84, 83, 118],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 31,
                "door_coordinates_list": [(61, 32), (62, 33)],
                "door_dp_indices_list": [119, 118],
                "entrance_in_coordinates_list": [(62, 32), (62, 33)],
                "entrance_out_coordinates_list": [(60, 32), (60, 33)],
                "entrance_in_dp_indices_list": [85, 86],
                "entrance_out_dp_indices_list": [116, 117],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            87: {
                "type": "passing_door",
                "location": (72, 25),
                "neighbors": [75, 93, 88],
                "within_room_name": "Rear Bridge",
                "color": "",
                "door_index": 9,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [9],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            88: {
                "type": "entrance_out",
                "location": (74, 25),
                "neighbors": [87, 89],
                "within_room_name": "Rear Bridge",
                "color": "",
                "door_index": 9,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Room 101",
            },
            89: {
                "type": "door",
                "location": (75, 25),
                "neighbors": [88, 90],
                "within_room_name": "Room 101",
                "color": "",
                "door_index": 9,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Room 101",
            },
            90: {
                "type": "entrance_in",
                "location": (76, 25),
                "neighbors": [89, 91],
                "within_room_name": "Room 101",
                "color": "",
                "door_index": 9,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Rear Bridge",
                "entering_room_name": "Room 101",
            },
            91: {
                "type": "room_corner",
                "location": (82, 25),
                "neighbors": [90, 92],
                "within_room_name": "Room 101",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            92: {
                "type": "victim",
                "location": (78, 21),
                "neighbors": [91],
                "within_room_name": "Room 101",
                "color": "yellow",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            93: {
                "type": "intersection",
                "location": (72, 37),
                "neighbors": [87, 105, 94],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            94: {
                "type": "passing_door",
                "location": (76, 37),
                "neighbors": [93, 95, 100],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [10, 11],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            95: {
                "type": "entrance_out",
                "location": (76, 36),
                "neighbors": [96, 94],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 10,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            96: {
                "type": "door",
                "location": (76, 35),
                "neighbors": [97, 95],
                "within_room_name": "Room 102",
                "color": "",
                "door_index": 10,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 102",
            },
            97: {
                "type": "entrance_in",
                "location": (76, 34),
                "neighbors": [98, 96],
                "within_room_name": "Room 102",
                "color": "",
                "door_index": 10,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 102",
            },
            98: {
                "type": "room_corner",
                "location": (76, 27),
                "neighbors": [97, 99],
                "within_room_name": "Room 102",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            99: {
                "type": "victim",
                "location": (82, 33),
                "neighbors": [98],
                "within_room_name": "Room 102",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            100: {
                "type": "entrance_out",
                "location": (76, 39),
                "neighbors": [94, 101],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 11,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 103",
            },
            101: {
                "type": "door",
                "location": (76, 40),
                "neighbors": [100, 102],
                "within_room_name": "Room 103",
                "color": "",
                "door_index": 11,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 103",
            },
            102: {
                "type": "entrance_in",
                "location": (76, 41),
                "neighbors": [101, 103],
                "within_room_name": "Room 103",
                "color": "",
                "door_index": 11,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 103",
            },
            103: {
                "type": "room_corner",
                "location": (76, 47),
                "neighbors": [102, 104],
                "within_room_name": "Room 103",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            104: {
                "type": "victim",
                "location": (81, 48),
                "neighbors": [103],
                "within_room_name": "Room 103",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            105: {
                "type": "passing_door",
                "location": (67, 37),
                "neighbors": [93, 109, 207],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 12,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [12],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            106: {
                "type": "door",
                "location": (67, 40),
                "neighbors": [207, 107],
                "within_room_name": "Room 104",
                "color": "",
                "door_index": 12,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 104",
            },
            107: {
                "type": "entrance_in",
                "location": (67, 41),
                "neighbors": [106, 108],
                "within_room_name": "Room 104",
                "color": "",
                "door_index": 12,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 104",
            },
            108: {
                "type": "room_center",
                "location": (67, 47),
                "neighbors": [107],
                "within_room_name": "Room 104",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            109: {
                "type": "intersection, passing_door",
                "location": (58, 37),
                "neighbors": [110, 115, 105, 175],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 13,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [13],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            110: {
                "type": "entrance_out",
                "location": (58, 39),
                "neighbors": [109, 111],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 13,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 105",
            },
            111: {
                "type": "door",
                "location": (58, 40),
                "neighbors": [110, 112],
                "within_room_name": "Room 105",
                "color": "",
                "door_index": 13,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 105",
            },
            112: {
                "type": "entrance_in",
                "location": (58, 41),
                "neighbors": [111, 113],
                "within_room_name": "Room 105",
                "color": "",
                "door_index": 13,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 105",
            },
            113: {
                "type": "room_corner",
                "location": (58, 47),
                "neighbors": [112, 114],
                "within_room_name": "Room 105",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            114: {
                "type": "victim",
                "location": (63, 47),
                "neighbors": [113],
                "within_room_name": "Room 105",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            115: {
                "type": "passing_door",
                "location": (58, 33),
                "neighbors": [116, 117, 120, 109],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 31,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [31],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            116: {
                "type": "entrance_out",
                "location": (60, 32),
                "neighbors": [119, 115],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 31,
                "door_coordinates_list": [(61, 32), (62, 33)],
                "door_dp_indices_list": [119, 118],
                "entrance_in_coordinates_list": [(62, 32), (62, 33)],
                "entrance_out_coordinates_list": [(60, 32), (60, 33)],
                "entrance_in_dp_indices_list": [85, 86],
                "entrance_out_dp_indices_list": [116, 117],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            117: {
                "type": "entrance_out",
                "location": (60, 33),
                "neighbors": [118, 115],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 31,
                "door_coordinates_list": [(61, 32), (62, 33)],
                "door_dp_indices_list": [119, 118],
                "entrance_in_coordinates_list": [(62, 32), (62, 33)],
                "entrance_out_coordinates_list": [(60, 32), (60, 33)],
                "entrance_in_dp_indices_list": [85, 86],
                "entrance_out_dp_indices_list": [116, 117],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            118: {
                "type": "door",
                "location": (61, 33),
                "neighbors": [86, 117],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 31,
                "door_coordinates_list": [(61, 32), (62, 33)],
                "door_dp_indices_list": [119, 118],
                "entrance_in_coordinates_list": [(62, 32), (62, 33)],
                "entrance_out_coordinates_list": [(60, 32), (60, 33)],
                "entrance_in_dp_indices_list": [85, 86],
                "entrance_out_dp_indices_list": [116, 117],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            119: {
                "type": "door",
                "location": (61, 32),
                "neighbors": [85, 116],
                "within_room_name": "Herbalife Conference Room",
                "color": "",
                "door_index": 31,
                "door_coordinates_list": [(61, 32), (62, 33)],
                "door_dp_indices_list": [119, 118],
                "entrance_in_coordinates_list": [(62, 32), (62, 33)],
                "entrance_out_coordinates_list": [(60, 32), (60, 33)],
                "entrance_in_dp_indices_list": [85, 86],
                "entrance_out_dp_indices_list": [116, 117],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Herbalife Conference Room",
            },
            120: {
                "type": "passing_door",
                "location": (58, 26),
                "neighbors": [115, 124, 125, 121],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [(56, 25), (56, 26)],
                "door_dp_indices_list": [126, 127],
                "entrance_in_coordinates_list": [(55, 25), (55, 26)],
                "entrance_out_coordinates_list": [(57, 25), (57, 26)],
                "entrance_in_dp_indices_list": [132, 133],
                "entrance_out_dp_indices_list": [124, 125],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Mary Kay Conference Room",
            },
            121: {
                "type": "passing_door",
                "location": (58, 23),
                "neighbors": [60, 122, 123, 120],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [(56, 22), (56, 23)],
                "door_dp_indices_list": [128, 129],
                "entrance_in_coordinates_list": [(55, 22), (55, 23)],
                "entrance_out_coordinates_list": [(57, 22), (57, 23)],
                "entrance_in_dp_indices_list": [130, 131],
                "entrance_out_dp_indices_list": [122, 123],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Amway Conference Room",
            },
            122: {
                "type": "entrance_out",
                "location": (57, 22),
                "neighbors": [121, 129],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [(56, 22), (56, 23)],
                "door_dp_indices_list": [128, 129],
                "entrance_in_coordinates_list": [(55, 22), (55, 23)],
                "entrance_out_coordinates_list": [(57, 22), (57, 23)],
                "entrance_in_dp_indices_list": [130, 131],
                "entrance_out_dp_indices_list": [122, 123],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Amway Conference Room",
            },
            123: {
                "type": "entrance_out",
                "location": (57, 23),
                "neighbors": [121, 128],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [(56, 22), (56, 23)],
                "door_dp_indices_list": [128, 129],
                "entrance_in_coordinates_list": [(55, 22), (55, 23)],
                "entrance_out_coordinates_list": [(57, 22), (57, 23)],
                "entrance_in_dp_indices_list": [130, 131],
                "entrance_out_dp_indices_list": [122, 123],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Amway Conference Room",
            },
            124: {
                "type": "entrance_out",
                "location": (57, 25),
                "neighbors": [120, 127],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [(56, 25), (56, 26)],
                "door_dp_indices_list": [126, 127],
                "entrance_in_coordinates_list": [(55, 25), (55, 26)],
                "entrance_out_coordinates_list": [(57, 25), (57, 26)],
                "entrance_in_dp_indices_list": [132, 133],
                "entrance_out_dp_indices_list": [124, 125],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Mary Kay Conference Room",
            },
            125: {
                "type": "entrance_out",
                "location": (57, 26),
                "neighbors": [120, 126],
                "within_room_name": "Middle Bridge",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [(56, 25), (56, 26)],
                "door_dp_indices_list": [126, 127],
                "entrance_in_coordinates_list": [(55, 25), (55, 26)],
                "entrance_out_coordinates_list": [(57, 25), (57, 26)],
                "entrance_in_dp_indices_list": [132, 133],
                "entrance_out_dp_indices_list": [124, 125],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Mary Kay Conference Room",
            },
            126: {
                "type": "door",
                "location": (56, 26),
                "neighbors": [125, 133],
                "within_room_name": "Mary Kay Conference Room",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [(56, 25), (56, 26)],
                "door_dp_indices_list": [126, 127],
                "entrance_in_coordinates_list": [(55, 25), (55, 26)],
                "entrance_out_coordinates_list": [(57, 25), (57, 26)],
                "entrance_in_dp_indices_list": [132, 133],
                "entrance_out_dp_indices_list": [124, 125],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Mary Kay Conference Room",
            },
            127: {
                "type": "door",
                "location": (56, 25),
                "neighbors": [124, 132],
                "within_room_name": "Mary Kay Conference Room",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [(56, 25), (56, 26)],
                "door_dp_indices_list": [126, 127],
                "entrance_in_coordinates_list": [(55, 25), (55, 26)],
                "entrance_out_coordinates_list": [(57, 25), (57, 26)],
                "entrance_in_dp_indices_list": [132, 133],
                "entrance_out_dp_indices_list": [124, 125],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Mary Kay Conference Room",
            },
            128: {
                "type": "door",
                "location": (56, 23),
                "neighbors": [123, 131],
                "within_room_name": "Amway Conference Room",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [(56, 22), (56, 23)],
                "door_dp_indices_list": [128, 129],
                "entrance_in_coordinates_list": [(55, 22), (55, 23)],
                "entrance_out_coordinates_list": [(57, 22), (57, 23)],
                "entrance_in_dp_indices_list": [130, 131],
                "entrance_out_dp_indices_list": [122, 123],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Amway Conference Room",
            },
            129: {
                "type": "door",
                "location": (56, 22),
                "neighbors": [122, 130],
                "within_room_name": "Amway Conference Room",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [(56, 22), (56, 23)],
                "door_dp_indices_list": [128, 129],
                "entrance_in_coordinates_list": [(55, 22), (55, 23)],
                "entrance_out_coordinates_list": [(57, 22), (57, 23)],
                "entrance_in_dp_indices_list": [130, 131],
                "entrance_out_dp_indices_list": [122, 123],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Amway Conference Room",
            },
            130: {
                "type": "entrance_in",
                "location": (55, 22),
                "neighbors": [129, 134],
                "within_room_name": "Amway Conference Room",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [(56, 22), (56, 23)],
                "door_dp_indices_list": [128, 129],
                "entrance_in_coordinates_list": [(55, 22), (55, 23)],
                "entrance_out_coordinates_list": [(57, 22), (57, 23)],
                "entrance_in_dp_indices_list": [130, 131],
                "entrance_out_dp_indices_list": [122, 123],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Amway Conference Room",
            },
            131: {
                "type": "entrance_in",
                "location": (55, 23),
                "neighbors": [128, 134],
                "within_room_name": "Amway Conference Room",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [(56, 22), (56, 23)],
                "door_dp_indices_list": [128, 129],
                "entrance_in_coordinates_list": [(55, 22), (55, 23)],
                "entrance_out_coordinates_list": [(57, 22), (57, 23)],
                "entrance_in_dp_indices_list": [130, 131],
                "entrance_out_dp_indices_list": [122, 123],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Amway Conference Room",
            },
            132: {
                "type": "entrance_in",
                "location": (55, 25),
                "neighbors": [127, 135],
                "within_room_name": "Mary Kay Conference Room",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [(56, 25), (56, 26)],
                "door_dp_indices_list": [126, 127],
                "entrance_in_coordinates_list": [(55, 25), (55, 26)],
                "entrance_out_coordinates_list": [(57, 25), (57, 26)],
                "entrance_in_dp_indices_list": [132, 133],
                "entrance_out_dp_indices_list": [124, 125],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Mary Kay Conference Room",
            },
            133: {
                "type": "entrance_in",
                "location": (55, 26),
                "neighbors": [126, 135],
                "within_room_name": "Mary Kay Conference Room",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [(56, 25), (56, 26)],
                "door_dp_indices_list": [126, 127],
                "entrance_in_coordinates_list": [(55, 25), (55, 26)],
                "entrance_out_coordinates_list": [(57, 25), (57, 26)],
                "entrance_in_dp_indices_list": [132, 133],
                "entrance_out_dp_indices_list": [124, 125],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "Middle Bridge",
                "entering_room_name": "Mary Kay Conference Room",
            },
            134: {
                "type": "victim",
                "location": (48, 23),
                "neighbors": [130, 131],
                "within_room_name": "Amway Conference Room",
                "color": "yellow",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            135: {
                "type": "victim",
                "location": (48, 34),
                "neighbors": [132, 133],
                "within_room_name": "Mary Kay Conference Room",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            136: {
                "type": "entrance_out",
                "location": (44, 33),
                "neighbors": [137, 139],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 28,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Women\'s Room",
                "entering_room_name": "Women\'s Stall 2",
            },
            137: {
                "type": "entrance_out",
                "location": (44, 29),
                "neighbors": [136, 138, 143],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 27,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Women\'s Room",
                "entering_room_name": "Women\'s Stall 1",
            },
            138: {
                "type": "door",
                "location": (43, 29),
                "neighbors": [137, 141],
                "within_room_name": "Women\'s Stall 1",
                "color": "",
                "door_index": 27,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Women\'s Room",
                "entering_room_name": "Women\'s Stall 1",
            },
            139: {
                "type": "door",
                "location": (43, 33),
                "neighbors": [140, 136],
                "within_room_name": "Women\'s Stall 2",
                "color": "",
                "door_index": 28,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Women\'s Room",
                "entering_room_name": "Women\'s Stall 2",
            },
            140: {
                "type": "entrance_in",
                "location": (42, 33),
                "neighbors": [142, 139],
                "within_room_name": "Women\'s Stall 2",
                "color": "",
                "door_index": 28,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Women\'s Room",
                "entering_room_name": "Women\'s Stall 2",
            },
            141: {
                "type": "entrance_in",
                "location": (42, 29),
                "neighbors": [138],
                "within_room_name": "Women\'s Stall 1",
                "color": "",
                "door_index": 27,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Women\'s Room",
                "entering_room_name": "Women\'s Stall 1",
            },
            142: {
                "type": "victim",
                "location": (40, 33),
                "neighbors": [140],
                "within_room_name": "Women\'s Stall 2",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            143: {
                "type": "entrance_in",
                "location": (40, 27),
                "neighbors": [144, 137],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 26,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Entrance Bridge",
                "entering_room_name": "Women\'s Room",
            },
            144: {
                "type": "door",
                "location": (39, 27),
                "neighbors": [143, 145],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 26,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Entrance Bridge",
                "entering_room_name": "Women\'s Room",
            },
            145: {
                "type": "entrance_out",
                "location": (38, 27),
                "neighbors": [144, 146],
                "within_room_name": "Entrance Bridge",
                "color": "",
                "door_index": 26,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Entrance Bridge",
                "entering_room_name": "Women\'s Room",
            },
            146: {
                "type": "passing_door",
                "location": (36, 27),
                "neighbors": [147, 145, 173],
                "within_room_name": "Entrance Bridge",
                "color": "",
                "door_index": 26,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [26],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            147: {
                "type": "passing_door",
                "location": (36, 18),
                "neighbors": [46, 146, 148],
                "within_room_name": "Entrance Bridge",
                "color": "",
                "door_index": 23,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [23],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            148: {
                "type": "entrance_out",
                "location": (38, 18),
                "neighbors": [147, 149],
                "within_room_name": "Entrance Bridge",
                "color": "",
                "door_index": 23,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Entrance Bridge",
                "entering_room_name": "Men\'s Room",
            },
            149: {
                "type": "door",
                "location": (39, 18),
                "neighbors": [148, 150],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 23,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Entrance Bridge",
                "entering_room_name": "Men\'s Room",
            },
            150: {
                "type": "entrance_in",
                "location": (40, 18),
                "neighbors": [149, 151],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 23,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Entrance Bridge",
                "entering_room_name": "Men\'s Room",
            },
            151: {
                "type": "entrance_out",
                "location": (44, 20),
                "neighbors": [150, 152],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 24,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Men\'s Room",
                "entering_room_name": "Men\'s Stall 1",
            },
            152: {
                "type": "door",
                "location": (43, 20),
                "neighbors": [153, 151],
                "within_room_name": "Men\'s Stall 1",
                "color": "",
                "door_index": 24,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Men\'s Room",
                "entering_room_name": "Men\'s Stall 1",
            },
            153: {
                "type": "entrance_in",
                "location": (42, 18),
                "neighbors": [152, 154],
                "within_room_name": "Men\'s Stall 1",
                "color": "",
                "door_index": 24,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Men\'s Room",
                "entering_room_name": "Men\'s Stall 1",
            },
            154: {
                "type": "victim",
                "location": (40, 20),
                "neighbors": [153],
                "within_room_name": "Men\'s Stall 1",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            155: {
                "type": "entrance_out",
                "location": (44, 24),
                "neighbors": [151, 156],
                "within_room_name": "Men\'s Stall 2",
                "color": "",
                "door_index": 25,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Men\'s Room",
                "entering_room_name": "Men\'s Stall 2",
            },
            156: {
                "type": "door",
                "location": (43, 24),
                "neighbors": [155, 157],
                "within_room_name": "Men\'s Stall 2",
                "color": "",
                "door_index": 25,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Men\'s Room",
                "entering_room_name": "Men\'s Stall 2",
            },
            157: {
                "type": "entrance_in",
                "location": (42, 24),
                "neighbors": [156],
                "within_room_name": "Men\'s Stall 2",
                "color": "",
                "door_index": 25,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "Men\'s Room",
                "entering_room_name": "Men\'s Stall 2",
            },
            158: {
                "type": "entrance_in",
                "location": (41, 14),
                "neighbors": [159],
                "within_room_name": "Den",
                "color": "",
                "door_index": 22,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Den",
            },
            159: {
                "type": "door",
                "location": (41, 13),
                "neighbors": [158, 160],
                "within_room_name": "Den",
                "color": "",
                "door_index": 22,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Den",
            },
            160: {
                "type": "entrance_out",
                "location": (41, 12),
                "neighbors": [159, 54],
                "within_room_name": "North Hallway",
                "color": "",
                "door_index": 22,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "North Hallway",
                "entering_room_name": "Den",
            },
            161: {
                "type": "room_corner",
                "location": (33, 14),
                "neighbors": [30, 167, 163, 162, 165],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            162: {
                "type": "room_corner",
                "location": (17, 14),
                "neighbors": [30, 164, 161, 165],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            163: {
                "type": "room_corner",
                "location": (33, 34),
                "neighbors": [168, 167, 161, 166, 165, 164],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            164: {
                "type": "room_corner",
                "location": (17, 34),
                "neighbors": [162, 168, 163, 165, 166],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            165: {
                "type": "room_center",
                "location": (25, 23),
                "neighbors": [166, 30, 161, 161, 164, 168, 167, 163],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            166: {
                "type": "victim",
                "location": (22, 25),
                "neighbors": [165, 164, 168, 163, 167],
                "within_room_name": "The Computer Farm",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            167: {
                "type": "victim",
                "location": (31, 31),
                "neighbors": [163, 161, 166, 167],
                "within_room_name": "The Computer Farm",
                "color": "yellow",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            168: {
                "type": "entrance_in",
                "location": (25, 34),
                "neighbors": [164, 163, 165, 166, 169],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": 20,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "The Computer Farm",
            },
            169: {
                "type": "door",
                "location": (25, 35),
                "neighbors": [168, 170],
                "within_room_name": "The Computer Farm",
                "color": "",
                "door_index": 20,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "The Computer Farm",
            },
            170: {
                "type": "entrance_out",
                "location": (25, 36),
                "neighbors": [169, 171],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 20,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "The Computer Farm",
            },
            171: {
                "type": "passing_door",
                "location": (25, 37),
                "neighbors": [190, 170, 172],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 20,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [20],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "The Computer Farm",
            },
            172: {
                "type": "passing_door",
                "location": (31, 37),
                "neighbors": [171, 173, 185],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 16,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [16],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 108",
            },
            173: {
                "type": "intersection",
                "location": (36, 37),
                "neighbors": [172, 174, 146],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            174: {
                "type": "passing_door",
                "location": (40, 37),
                "neighbors": [173, 175, 180],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 15,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [15],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 107",
            },
            175: {
                "type": "passing_door",
                "location": (49, 37),
                "neighbors": [174, 109, 176],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 14,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [14],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 106",
            },
            176: {
                "type": "entrance_out",
                "location": (49, 39),
                "neighbors": [175, 177],
                "within_room_name": "Room 106",
                "color": "",
                "door_index": 14,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 106",
            },
            177: {
                "type": "door",
                "location": (49, 40),
                "neighbors": [176, 178],
                "within_room_name": "Room 106",
                "color": "",
                "door_index": 14,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 106",
            },
            178: {
                "type": "entrance_in",
                "location": (49, 41),
                "neighbors": [177, 179],
                "within_room_name": "Room 106",
                "color": "",
                "door_index": 14,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 106",
            },
            179: {
                "type": "room_corner",
                "location": (49, 47),
                "neighbors": [178],
                "within_room_name": "Room 106",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            180: {
                "type": "entrance_out",
                "location": (40, 39),
                "neighbors": [174, 181],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 15,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 107",
            },
            181: {
                "type": "door",
                "location": (40, 40),
                "neighbors": [180, 182],
                "within_room_name": "Room 107",
                "color": "",
                "door_index": 15,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 107",
            },
            182: {
                "type": "entrance_in",
                "location": (40, 41),
                "neighbors": [181, 183],
                "within_room_name": "Room 107",
                "color": "",
                "door_index": 15,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 107",
            },
            183: {
                "type": "room_corner",
                "location": (40, 47),
                "neighbors": [182, 184],
                "within_room_name": "Room 107",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            184: {
                "type": "victim",
                "location": (45, 47),
                "neighbors": [183],
                "within_room_name": "Room 107",
                "color": "yellow",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            185: {
                "type": "entrance_out",
                "location": (31, 39),
                "neighbors": [172, 186],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 16,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 108",
            },
            186: {
                "type": "door",
                "location": (31, 40),
                "neighbors": [185, 187],
                "within_room_name": "Room 108",
                "color": "",
                "door_index": 16,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 108",
            },
            187: {
                "type": "entrance_in",
                "location": (31, 41),
                "neighbors": [186, 188],
                "within_room_name": "Room 108",
                "color": "",
                "door_index": 16,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 108",
            },
            188: {
                "type": "room_corner",
                "location": (31, 47),
                "neighbors": [187, 189],
                "within_room_name": "Room 108",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            189: {
                "type": "victim",
                "location": (37, 47),
                "neighbors": [188],
                "within_room_name": "Room 108",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            190: {
                "type": "passing_door",
                "location": (22, 37),
                "neighbors": [171, 196, 191],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 17,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [17],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 109",
            },
            191: {
                "type": "entrance_out",
                "location": (22, 39),
                "neighbors": [190, 192],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 17,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 109",
            },
            192: {
                "type": "door",
                "location": (22, 40),
                "neighbors": [191, 193],
                "within_room_name": "Room 109",
                "color": "",
                "door_index": 17,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 109",
            },
            193: {
                "type": "entrance_in",
                "location": (22, 41),
                "neighbors": [192, 194],
                "within_room_name": "Room 109",
                "color": "",
                "door_index": 17,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 109",
            },
            194: {
                "type": "room_corner",
                "location": (22, 47),
                "neighbors": [193, 195],
                "within_room_name": "Room 109",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            195: {
                "type": "victim",
                "location": (27, 43),
                "neighbors": [194],
                "within_room_name": "Room 109",
                "color": "green",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            196: {
                "type": "passing_door",
                "location": (13, 37),
                "neighbors": [190, 202, 197],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 18,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [18],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 110",
            },
            197: {
                "type": "entrance_out",
                "location": (13, 39),
                "neighbors": [196, 198],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 18,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [18],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 110",
            },
            198: {
                "type": "door",
                "location": (13, 40),
                "neighbors": [197, 199],
                "within_room_name": "Room 110",
                "color": "",
                "door_index": 18,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [18],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 110",
            },
            199: {
                "type": "entrance_in",
                "location": (13, 41),
                "neighbors": [198, 200],
                "within_room_name": "Room 110",
                "color": "",
                "door_index": 18,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [18],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 110",
            },
            200: {
                "type": "room_corner",
                "location": (13, 47),
                "neighbors": [199, 201],
                "within_room_name": "Room 110",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            201: {
                "type": "victim",
                "location": (19, 47),
                "neighbors": [200],
                "within_room_name": "Room 110",
                "color": "yellow",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            202: {
                "type": "passing_door",
                "location": (4, 37),
                "neighbors": [196, 203],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 19,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [19],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 111",
            },
            203: {
                "type": "entrance_out",
                "location": (4, 39),
                "neighbors": [202, 204],
                "within_room_name": "South Hallway",
                "color": "",
                "door_index": 19,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 111",
            },
            204: {
                "type": "door",
                "location": (4, 40),
                "neighbors": [203, 205],
                "within_room_name": "Room 111",
                "color": "",
                "door_index": 19,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 111",
            },
            205: {
                "type": "entrance_in",
                "location": (4, 41),
                "neighbors": [204, 206],
                "within_room_name": "Room 111",
                "color": "",
                "door_index": 19,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 111",
            },
            206: {
                "type": "room_corner",
                "location": (4, 47),
                "neighbors": [205],
                "within_room_name": "Room 111",
                "color": "",
                "door_index": None,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            207: {
                "type": "entrance_out",
                "location": (67, 39),
                "neighbors": [105, 106],
                "within_room_name": "Room 104",
                "color": "",
                "door_index": 12,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [],
                "exiting_room_name": "South Hallway",
                "entering_room_name": "Room 104",
            },

        }
        return all_decision_points_xy

    def update_victims_record(self, victim_save_record):
        if self.num_saved != len(victim_save_record):
            for index in victim_save_record:
                loc = (victim_save_record[index]['x'], victim_save_record[index]['y'])
                self.victim_path_details[self.location_to_victim_path_idx[loc]]['saved_state'] = True

    def check_room(self, pos_x, pos_y):
        if (pos_x, pos_y) in self.reversed_complete_room_dict:
            room_to_go = self.reversed_complete_room_dict[(pos_x, pos_y)]
        else:
            room_to_go = "undefined"
        return room_to_go

    def get_room(self, pos_x, pos_y):
        if (pos_x, pos_y) in self.reversed_complete_room_dict:
            room_to_go = self.reversed_complete_room_dict[(pos_x, pos_y)]
            if room_to_go not in self.visited_rooms_list:
                self.visited_rooms_list.append(room_to_go)
        else:
            room_to_go = "undefined"
        return room_to_go

    def generate_victim_path_details(self):
        self.num_saved = 0
        self.victim_path_details = {
            0: {
                "id": 0,
                "location": (30, 4),
                'color': "green",
                'saved_state': False,
            },
            1: {
                "id": 1,
                "location": (35, 7),
                'color': "yellow",
                'saved_state': False,

            },
            2: {
                "id": 14,
                "location": (22, 25),
                'color': "green",
                'saved_state': False,
            },
            3: {
                "id": 15,
                "location": (31, 31),
                'color': "yellow",
                'saved_state': False,
            },
            4: {
                "id": 12,
                "location": (27, 43),
                'color': "green",
                'saved_state': False,
            },
            5: {
                "id": 13,
                "location": (19, 47),
                'color': "yellow",
                'saved_state': False,
            },
            6: {
                "id": 10,
                "location": (45, 47),
                'color': "yellow",
                'saved_state': False,
            },
            7: {
                "id": 21,
                "location": (65, 30),
                'color': "yellow",
                'saved_state': False,
            },
            8: {
                "id": 20,
                "location": (67, 18),
                'color': "yellow",
                'saved_state': False,
            },
            9: {
                "id": 6,
                "location": (78, 21),
                'color': "yellow",
                'saved_state': False,
            },
            10: {
                "id": 18,
                "location": (48, 23),
                'color': "yellow",
                'saved_state': False,
            },
            11: {
                "id": 19,
                "location": (48, 34),
                'color': "green",
                'saved_state': False,
            },
            12: {
                "id": 9,
                "location": (63, 47),
                'color': "green",
                'saved_state': False,
            },
            13: {
                "id": 8,
                "location": (81, 48),
                'color': "green",
                'saved_state': False,
            },
            14: {
                "id": 7,
                "location": (82, 33),
                'color': "green",
                'saved_state': False,
            },
            15: {
                "id": 4,
                "location": (74, 6),
                'color': "green",
                'saved_state': False,
            },
            16: {
                "id": 5,
                "location": (87, 4),
                'color': "green",
                'saved_state': False,
            },
            17: {
                "id": 3,
                "location": (48, 6),
                'color': "green",
                'saved_state': False,
            },
            18: {
                "id": 2,
                "location": (50, 5),
                'color': "green",
                'saved_state': False,
            },
            19: {
                "id": 17,
                "location": (40, 20),
                'color': "green",
                'saved_state': False,
            },
            20: {
                "id": 16,
                "location": (40, 33),
                'color': "green",
                'saved_state': False,
            },
            21: {
                "id": 11,
                "location": (37, 47),
                'color': "green",
                'saved_state': False,
            },

        }
        self.location_to_victim_path_idx = {}
        for index in self.victim_path_details:
            self.location_to_victim_path_idx[self.victim_path_details[index]['location']] = index

    def decision_point_dict(self):
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
                'location': (30, 4),
                'victim_index': 0,
                'color': 'g',
            },
            70: {
                'type': 'victim',
                'location': (35, 7),
                'victim_index': 1,
                'color': 'y',
            },
            71: {
                'type': 'victim',
                'location': (50, 5),
                'victim_index': 2,
                'color': 'g',
            },
            72: {
                'type': 'victim',
                'location': (48, 6),
                'victim_index': 3,
                'color': 'g',
            },
            73: {
                'type': 'victim',
                'location': (74, 6),
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
                'location': (78, 21),
                'victim_index': 6,
                'color': 'y',
            },
            76: {
                'type': 'victim',
                'location': (82, 33),
                'victim_index': 7,
                'color': 'g',
            },
            77: {
                'type': 'victim',
                'location': (81, 48),
                'victim_index': 8,
                'color': 'g',
            },
            78: {
                'type': 'victim',
                'location': (63, 47),
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
            if decision_points[key]['type'] == 'intersection':
                intersections[key] = decision_points[key]
            if decision_points[key]['type'] == 'entrance':
                entrances[key] = decision_points[key]
            if decision_points[key]['type'] == 'door' or decision_points[key]['type'] == 'open door':
                doors[key] = decision_points[key]
        return decision_points, intersections, entrances, doors

    def make_doors_list(self):
        self.doors_dict = {0: (40, 4), 1: (40, 13), 2: (1, 16), 3: (2, 16), 4: (4, 16), 5: (5, 16), 6: (8, 25),
                           7: (13, 25),
                           8: (35, 25), 9: (40, 22), 10: (40, 31), 11: (8, 37), 12: (13, 41), 13: (18, 39),
                           14: (20, 43),
                           15: (24, 43), 16: (27, 39), 17: (29, 43), 18: (33, 43), 19: (40, 40), 20: (8, 54),
                           21: (22, 56),
                           22: (23, 56), 23: (25, 56), 24: (26, 56), 25: (32, 61), 26: (33, 61), 27: (40, 49),
                           28: (40, 58),
                           29: (8, 69), 30: (8, 70), 31: (15, 70), 32: (16, 70), 33: (25, 75), 34: (35, 76),
                           35: (40, 67),
                           36: (40, 76), 37: (8, 84), 38: (8, 30), 39: (8, 31), 40: (8, 32), 41: (8, 33)}
        with open(prefix + "setup_files/door_details_dict.pkl", 'rb') as filename:
            self.doors_details_dict = pkl.load(filename)

    def get_direction_of_movement(self, start, end):
        x_diff = end[0] - start[0]
        y_diff = end[1] - start[1]
        final_direction = None
        if abs(x_diff) > abs(y_diff):
            if x_diff >= 0:
                final_direction = EAST
            else:
                final_direction = WEST
        else:
            if y_diff >= 0:
                final_direction = SOUTH
            else:
                final_direction = NORTH
        return final_direction

    def get_door_side(self, curr_location, curr_heading, door_locations):
        right = 0
        left = 0
        for i in range(len(door_locations)):
            y_diff = door_locations[i][0] - curr_location[0]  # change in columns
            x_diff = door_locations[i][1] - curr_location[1]  # change in rows
            if curr_heading == NORTH:
                if x_diff > 0:
                    right += 1
                elif x_diff < 0:
                    left += 1
            elif curr_heading == EAST:
                if y_diff > 0:
                    right += 1
                elif y_diff < 0:
                    left += 1
            elif curr_heading == SOUTH:
                if x_diff < 0:
                    right += 1
                elif x_diff > 0:
                    left += 1
            elif curr_heading == WEST:
                if y_diff < 0:
                    right += 1
                elif y_diff > 0:
                    left += 1

        if right > left:
            return RIGHT
        if left > left:
            return LEFT
        return AHEAD

    def get_door_side_one_door_coord(self, curr_location, curr_heading, door_location):
        # print("curr_heading", curr_heading)
        # print("input to door side", (curr_location, door_location))
        door_side = None
        x_diff = door_location[0] - curr_location[0]  # change in columns
        y_diff = door_location[1] - curr_location[1]  # change in rows
        # print("(x_diff, y_diff)", (x_diff, y_diff))

        if abs(x_diff) > abs(y_diff):
            if curr_heading == NORTH:
                if x_diff > 0:
                    door_side = RIGHT
                elif x_diff < 0:
                    door_side = LEFT
            elif curr_heading == SOUTH:
                if x_diff < 0:
                    door_side = RIGHT
                elif x_diff > 0:
                    door_side = LEFT
            elif curr_heading == EAST:
                if x_diff > 0:
                    door_side = AHEAD
                elif x_diff < 0:
                    door_side = BEHIND
            elif curr_heading == WEST:
                if x_diff < 0:
                    door_side = AHEAD
                elif x_diff > 0:
                    door_side = BEHIND

        elif abs(y_diff) >= abs(x_diff):
            if curr_heading == EAST:
                if y_diff > 0:
                    door_side = RIGHT
                elif y_diff < 0:
                    door_side = LEFT
            elif curr_heading == WEST:
                if y_diff < 0:
                    door_side = RIGHT
                elif y_diff > 0:
                    door_side = LEFT
            elif curr_heading == NORTH:
                if y_diff < 0:
                    door_side = AHEAD
                elif y_diff > 0:
                    door_side = BEHIND
            elif curr_heading == SOUTH:
                if y_diff > 0:
                    door_side = AHEAD
                elif y_diff < 0:
                    door_side = BEHIND

        # print("found door side: ", self.turn_advice_dict[door_side])
        # print()
        return door_side

    def get_turn_direction(self, prev_dir, next_dir):
        turn_direction = None
        if prev_dir == NORTH:
            if next_dir == EAST:
                turn_direction = RIGHT
            if next_dir == WEST:
                turn_direction = LEFT
            if next_dir == SOUTH:
                turn_direction = AROUND

        if prev_dir == SOUTH:
            if next_dir == EAST:
                turn_direction = LEFT
            if next_dir == WEST:
                turn_direction = RIGHT
            if next_dir == NORTH:
                turn_direction = AROUND

        if prev_dir == WEST:
            if next_dir == EAST:
                turn_direction = AROUND
            if next_dir == NORTH:
                turn_direction = RIGHT
            if next_dir == SOUTH:
                turn_direction = LEFT

        if prev_dir == EAST:
            if next_dir == WEST:
                turn_direction = AROUND
            if next_dir == NORTH:
                turn_direction = LEFT
            if next_dir == SOUTH:
                turn_direction = RIGHT

        return turn_direction


    def generate_level_3_instructions(self, start_location, end_location, curr_heading, victim_save_record,
                                                to_plot=False):
        num_yellows = 0
        num_greens = 0
        if end_location in self.yellow_locs:
            color = 'red'
            num_yellows = 1
        else:
            color = 'blue'
            num_greens = 1

        try:
            with Timeout(2):
                xy_path = recompute_path(start_location, end_location, self.id_to_goal_tuple,
                                         self.inverse_gameboard, self.flipped_obstacles_stairs_free, self.flipped_yellow_locs,
                                         self.flipped_green_locs)
        except Timeout.Timeout:
            xy_path = []

        # print("done computing\n")
        rc_path = [(j, i) for (i, j) in xy_path]

        rooms_list = []
        for (i, j) in xy_path:
            if (i, j) in self.reversed_complete_room_dict:
                room_to_go = self.reversed_complete_room_dict[(i, j)]
                if len(rooms_list) == 0 or rooms_list[-1] != room_to_go:
                    rooms_list.append(room_to_go)

        if len(rooms_list)==0:
            advice = ""
            return advice

        if len(rooms_list) <= 1:
            advice = "In " + rooms_list[0] + ", "

        else:
            advice = "Go from " + rooms_list[0]
            for room_name in rooms_list[1:]:
                advice += ", to " + room_name
            advice += ". "

        next_victim_idx = len(victim_save_record)
        next_victim_to_save_id = self.victim_path[next_victim_idx]

        if next_victim_idx + 1 < len(self.victim_path):
            in_same_room = True
            while in_same_room:
                next_victim_idx += 1
                next_victim_to_save_id = self.victim_path[next_victim_idx]
                check_next_location = self.id_to_goal_tuple[next_victim_to_save_id]
                next_room_to_go = self.reversed_complete_room_dict[check_next_location]
                if next_room_to_go == rooms_list[-1]:
                    if check_next_location in self.yellow_locs:
                        num_yellows += 1
                    else:
                        num_greens += 1
                else:
                    in_same_room = False

        if num_yellows > 1 and num_greens == 0:
            advice += "Triage " + str(num_yellows) + " red victims."
        elif num_yellows == 1 and num_greens == 0:
            advice += "Triage the red victim."
        elif num_greens > 1 and num_yellows == 0:
            advice += "Triage " + str(num_greens) + " blue victims."
        elif num_greens == 1 and num_yellows == 0:
            advice += "Triage the blue victim."
        elif num_yellows == 1 and num_greens == 1:
            if color == 'red':
                advice += "Triage the red victim first, and then the blue victim."
            else:
                advice += "Triage the blue victim first, and then the red victim."
        elif num_yellows > 1 and num_greens > 1:
            if color == 'red':
                advice += "Triage " + str(num_yellows) + " red victims first, and then " + str(
                    num_greens) + "blue victims."
            else:
                advice += "Triage " + str(num_greens) + " blue victims first, and then " + str(
                    num_yellows) + "red victims."

        else:
            advice += "Triage " + color + " victim."
        return advice


    def generate_level_1_instructions(self, start_location, end_location, curr_heading, victim_save_record,
                                                to_plot=False):
        # print('type:', type(victim_save_record))
        if end_location in self.yellow_locs:
            color = 'red'
            num_clicks = 1
        else:
            color = 'blue'
            num_clicks = 1
        # victim_save_record = json.loads(victim_save_record)
        # print('victim_save_record', victim_save_record)

        # print("self.victim_path", self.victim_path)
        # print("self.id_to_goal_tuple", self.id_to_goal_tuple)
        # print("yellow locs", self.yellow_locs)
        # print("green locs", self.green_locs)

        # if len(victim_save_record) > 0:
        #     max_key = max(list(victim_save_record.keys()))
        #     actual_start_location = (victim_save_record[max_key]['x'], victim_save_record[max_key]['y'])
        # else:
        #     actual_start_location = (6, 5)

        xy_path = recompute_path(start_location, end_location, self.id_to_goal_tuple,
                                 self.inverse_gameboard, self.flipped_obstacles_stairs_free, self.flipped_yellow_locs,
                                 self.flipped_green_locs)
        rc_path = [(j, i) for (i, j) in xy_path]
        advice_list = []

        prev_direction_of_movement = curr_heading
        current_movement_command = {
            'direction': prev_direction_of_movement,
            'count': 0,
        }
        # print("prev_direction_of_movement", prev_direction_of_movement)
        for i in range(2, len(xy_path)):
            (prev_x, prev_y) = xy_path[i - 1]
            (curr_x, curr_y) = xy_path[i]
            # print((prev_x, prev_y), (curr_x, curr_y))
            direction_of_movement = self.get_direction_of_movement((prev_x, prev_y), (curr_x, curr_y))
            # print("direction_of_movement", direction_of_movement)
            if direction_of_movement == prev_direction_of_movement:
                current_movement_command['count'] += 1
            else:
                turn_direction = self.get_turn_direction(prev_direction_of_movement, direction_of_movement)
                # print("turn_direction", turn_direction)
                if current_movement_command['count'] == 0:
                    advice_list.append("Turn " + self.turn_advice_dict[turn_direction] + ".")
                else:
                    advice_list.append("Walk forward " + str(current_movement_command['count']) + " steps.")
                    advice_list.append("Turn " + self.turn_advice_dict[turn_direction] + ".")

                current_movement_command = {
                    'direction': direction_of_movement,
                    'count': 1,
                }

            prev_direction_of_movement = direction_of_movement

        if current_movement_command['count'] > 0:
            advice_list.append("Walk forward " + str(current_movement_command['count']) + " steps.")
        advice_list.append("Press SPACE " + str(num_clicks) + "x at " + color + " victim location.")

        advice = "\n".join(advice_list[:min(2, len(advice_list))])
        # advice = " ".join(advice_list)
        return advice

    def compute_nearest_dp(self):
        self.dp_dict = self.generate_decision_points_augmented()
        distances_dict = {}
        for i in range(93):
            for j in range(50):
                if (j, i) in self.obstacles:
                    continue
                closest_dp_key = 0
                closest_dp_distance = 10000
                for end_key in self.dp_dict:
                    print("start_loc, end_key", ((i, j), end_key))

                    try:
                        with Timeout(5):
                            travel_path = recompute_path((i, j), self.dp_dict[end_key]['location'],
                                                         self.id_to_goal_tuple,
                                                         self.inverse_gameboard, self.flipped_obstacles,
                                                         self.flipped_yellow_locs,
                                                         self.flipped_green_locs)
                    except Timeout.Timeout:
                        travel_path = []

                    if len(travel_path) < closest_dp_distance:
                        closest_dp_distance = len(travel_path)
                        closest_dp_key = end_key
                    if len(travel_path) == 0:
                        break

                distances_dict[(i, j)] = closest_dp_key
        with open(prefix + "setup_files/shortest_augmented_dps_distances_2.pkl", 'wb') as filename:
            pkl.dump(distances_dict, filename)
        # print("distances_dict", distances_dict)

    def get_closest_idx_to_pt_manhattan(self, point, other_pts):
        closest_idx = 0
        closest_distance = 10000
        closest_location = None
        for i in range(len(other_pts)):
            dist = manhattan_dist(point[0], point[1], other_pts[i][0], other_pts[i][1])
            if dist < closest_distance:
                closest_distance = dist
                closest_idx = i
                closest_location = (other_pts[i][0], other_pts[i][1])
        return closest_idx, closest_location


    def generate_level_2_instructions(self, start_location, end_location, curr_heading, victim_save_record,
                                                past_traj_input,
                                                to_plot=False):



        if start_location not in self.closest_decision_pt:
            return '', None
        start_idx = self.closest_decision_pt[start_location]

        goal_idx = self.closest_decision_pt[end_location]
        if goal_idx == 43:
            goal_idx = 45

        decision_search = Decision_Point_Search_Augmented(self.dp_dict)
        travel_path = decision_search.a_star(start_idx, goal_idx)
        # print("travel_path", travel_path)

        advice_list = []
        types_list = []
        destination_dp_list = []

        for i in range(0, len(travel_path)):
            types_list.append(self.dp_dict[travel_path[i]]['type'])

        types_based_advice_list = [" "]
        path_heading = curr_heading
        door_side_count = {
            LEFT: 0,
            RIGHT: 0,
            AHEAD: 0,
            BEHIND: 0,
            AROUND: 0,
        }
        intersection_count = 0
        if travel_path[0] == 0:
            types_based_advice_list.append(" ")

        if len(travel_path) == 1:
            if self.dp_dict[travel_path[0]]['color'] == 'green':
                types_based_advice_list.append('Find and save BLUE victim in current room.')
            else:
                types_based_advice_list.append('Find and save RED victim in current room.')

        for i in range(len(types_list) - 1):
            if i == 0:
                next_direction = curr_heading
                if 'passing_door' in types_list[i] or 'intersection' in types_list[i]:
                    next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                                    self.dp_dict[travel_path[i + 1]]['location'])
                    if next_direction != curr_heading:

                        turn_direction = self.get_turn_direction(path_heading, next_direction)
                        if turn_direction is not None:
                            if types_list[i + 1] == 'entrance_out':
                                types_based_advice_list.append(
                                    "Turn " + self.turn_advice_dict[turn_direction] + ' at door.')
                            else:
                                types_based_advice_list.append(
                                    "Turn " + self.turn_advice_dict[turn_direction] + ' immediately.')
                            intersection_count = 0

                if types_list[i] == 'entrance_out' and 'passing_door' in types_list[i + 1]:
                    if i + 2 < len(types_list):
                        next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i + 1]]['location'],
                                                                        self.dp_dict[travel_path[i + 2]]['location'])
                    if next_direction != curr_heading:
                        turn_direction = self.get_turn_direction(path_heading, next_direction)
                        if turn_direction is not None:
                            types_based_advice_list.append(
                                "First, turn " + self.turn_advice_dict[turn_direction] + '.')
                            intersection_count = 0

            if types_list[i] == 'door':
                if types_list[i + 1] == 'entrance_in':
                    if i > 2 and [travel_path[i - 3], travel_path[i - 2], travel_path[i - 1], travel_path[i],
                                  travel_path[i + 1]] == [100, 94, 95, 96, 97]:
                        types_based_advice_list.append("Enter room directly across the hallway.")
                    elif i > 2 and [travel_path[i - 3], travel_path[i - 2], travel_path[i - 1], travel_path[i],
                                    travel_path[i + 1]] == [115, 109, 110, 111, 112]:
                        types_based_advice_list.append("Enter room directly at the end of the hallway.")
                    elif i< len(types_list) and self.dp_dict[travel_path[i + 1]]["within_room_name"] == "Kings Terrace":
                        types_based_advice_list.append("Enter terrace within room.")

                    elif len(types_based_advice_list) > 0 and types_based_advice_list[-1] == 'Exit room.':
                        types_based_advice_list.append("Enter next room ahead.")
                    else:
                        types_based_advice_list.append("Enter room.")

            if 'passing_door' in types_list[i] and i > 0:
                if types_list[i - 1] != 'entrance_out':
                    doors_passed = self.dp_dict[travel_path[i]]['passing_associated_door_indices']
                    for door_idx in doors_passed:
                        door_coordinates = [self.dp_dict[d_idx]['location'] for d_idx in
                                            self.door_index_to_dp_indices_dict[door_idx]]
                        closest_i, closest_door_location = self.get_closest_idx_to_pt_manhattan(
                            self.dp_dict[i]['location'], door_coordinates)
                        door_side = self.get_door_side_one_door_coord(self.dp_dict[travel_path[i]]['location'],
                                                                      path_heading, closest_door_location)
                        door_side_count[door_side] += 1
                        # print("door_side_count", door_side_count)

            if 'passing_door' in types_list[i] and types_list[i + 1] == 'entrance_out':

                next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                                self.dp_dict[travel_path[i + 1]]['location'])

                turn_direction = self.get_turn_direction(path_heading, next_direction)

                if turn_direction is not None:
                    if len(types_based_advice_list) > 0 and 'at door' in types_based_advice_list[-1]:
                        skip = True
                    else:
                        types_based_advice_list.append("Proceed to " + self.nth_advice_dict[door_side_count[turn_direction]]
                                                       + " door " + self.room_side_turn_advice_dict[turn_direction] + ".")

                # if turn_direction is not None and "Turn" not in types_based_advice_list[-1]:
                #     types_based_advice_list.append("Turn "+self.turn_advice_dict[turn_direction]+' to enter.')

                door_side_count = {
                    LEFT: 0,
                    RIGHT: 0,
                    AHEAD: 0,
                    BEHIND: 0,
                    AROUND: 0,
                }
                intersection_count = 0

            if types_list[i] == 'door':
                if types_list[i + 1] == 'entrance_out':
                    if travel_path[i] in [29, 77, 80]:
                        types_based_advice_list.append("Exit room out of North-side door.")
                    elif travel_path[i] in [169, 119, 118]:
                        types_based_advice_list.append("Exit room out of South-side door.")
                    else:
                        types_based_advice_list.append("Exit room.")

            if 'intersection' in types_list[i]:
                next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                                self.dp_dict[travel_path[i + 1]]['location'])
                # print("directions to go ", path_heading, next_direction)
                intersection_count += 1
                turn_direction = self.get_turn_direction(path_heading, next_direction)
                # print("turn_direction", turn_direction)
                if turn_direction is not None:
                    door_side_count = {
                        LEFT: 0,
                        RIGHT: 0,
                        AHEAD: 0,
                        BEHIND: 0,
                        AROUND: 0,
                    }
                    # if "Turn" not in types_based_advice_list[-1] and "turn" not in types_based_advice_list[-1]:
                    if len(types_based_advice_list) > 0 and 'intersection' in types_based_advice_list[-1]:
                        types_based_advice_list.append(
                            "At next intersection, turn " + self.turn_advice_dict[turn_direction] + ".")
                    else:
                        types_based_advice_list.append(
                            "At the " +self.nth_advice_dict[intersection_count]+ " intersection, turn " + self.turn_advice_dict[turn_direction] + ".")
                        intersection_count = 0

            if 'passing_door' in types_list[i] and i > 0:
                if types_list[i - 1] == 'entrance_out':
                    next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                                    self.dp_dict[travel_path[i + 1]]['location'])

                    if travel_path[i - 1] in [32, 33, 34, 35] and travel_path[i] == 31:
                        path_heading = SOUTH
                        # print(f'changing path heading to {path_heading}')

                    turn_direction = self.get_turn_direction(path_heading, next_direction)
                    # print(f'Path heading {path_heading} to next: {next_direction}, direction = {turn_direction}')
                    # print(f'on travel path i = {i}, {travel_path[i]}, full path = {travel_path}\n')

                    if turn_direction is not None:
                        # print("directions to go ", path_heading, next_direction)
                        if "Turn" not in types_based_advice_list[-1] and "turn" not in types_based_advice_list[-1]:
                            types_based_advice_list.append(
                                "Upon exit, turn " + self.turn_advice_dict[turn_direction] + " immediately.")

            path_heading = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                          self.dp_dict[travel_path[i + 1]]['location'])

            if types_list[i + 1] == 'victim':
                if self.dp_dict[travel_path[i + 1]]['color'] == 'green':
                    types_based_advice_list.append('Find and save BLUE victim in current room.')
                else:
                    if travel_path[i + 1] == 82:
                        types_based_advice_list.append('Find and save the second RED victim in current room.')
                    else:
                        types_based_advice_list.append('Find and save RED victim in current room.')

        # self.complexity_level += 1

        generated_advice = "\n".join(types_based_advice_list[:min(3, len(types_based_advice_list))])
        output = generated_advice
        # self.level_2_static_hold_counter += 1
        # if self.level_2_static_hold_counter >= 5:
        #     self.level_2_static_hold_counter = 0
        #     self.level_2_hold_advice = generated_advice
        #     output = generated_advice
        # else:
        #     output = self.level_2_hold_advice

        final_destination = None
        return output, final_destination

    def compute_likelihood_of_goal(self):
        return 1

    def check_player_floodfill_bayesian_efficient(self, past_trajectory, player_x, player_y, victim_idx,
                                                  previously_saved):
        # print("id_to_goal_tuple", self.id_to_goal_tuple)
        gameboard_floodfill_current = self.gameboard_floodfill_dictionary[victim_idx]
        epsilon_dist = euclidean((player_x, player_y), self.id_to_goal_tuple[victim_idx])
        if epsilon_dist < 5:
            epsilon = 0.1
        else:
            epsilon = 0.3
        beta = 0.8
        lookback = 5

        prob_goal = beta
        prob_non_goal = (1 - beta) / (len(self.id_to_goal_tuple) - len(previously_saved))
        denominator = 0
        # Get the probability of the current goal and trajectory
        # print('reversed_complete_room_dict', reversed_complete_room_dict)
        if len(past_trajectory) <= 1:
            for candidate_idx in self.gameboard_floodfill_dictionary.keys():
                if candidate_idx in previously_saved:
                    self.previous_likelihood_dict[candidate_idx] = [0] * 20
                else:
                    self.previous_likelihood_dict[candidate_idx] = [1] * 20
            if len(past_trajectory) < 2:
                return 1

        victim_likelihood_product = 0
        for candidate_idx in self.gameboard_floodfill_dictionary.keys():
            if candidate_idx in previously_saved:
                self.previous_likelihood_dict[candidate_idx] = [0] * 20
                continue

            # if candidate_idx == victim_idx and len(past_trajectory) < 2:
            #     self.previous_likelihood_dict[candidate_idx] = [1] * 20

            prob_traj_given_goal = 1
            gameboard_floodfill = self.gameboard_floodfill_dictionary[candidate_idx]
            # print("gameboard_floodfill.shape", gameboard_floodfill.shape)

            cur_loc = past_trajectory[-1]
            prev_loc = past_trajectory[-2]
            if gameboard_floodfill[cur_loc[0], cur_loc[1]] <= gameboard_floodfill[prev_loc[0], prev_loc[1]]:
                proba = 1 - epsilon

            else:
                proba = (epsilon)
            # print("proba = ", proba)
            count_denom = proba
            if cur_loc[0] - 1 >= 0 and gameboard_floodfill[cur_loc[0] - 1, cur_loc[1]] <= gameboard_floodfill[
                cur_loc[0], cur_loc[1]]:
                count_denom += 1 - epsilon
            if cur_loc[0] + 1 < gameboard_floodfill.shape[0] and gameboard_floodfill[cur_loc[0] + 1, cur_loc[1]] <= \
                    gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1 - epsilon
            if cur_loc[1] + 1 < gameboard_floodfill.shape[1] and gameboard_floodfill[cur_loc[0], cur_loc[1] + 1] <= \
                    gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1 - epsilon
            if cur_loc[1] - 1 >= 0 and gameboard_floodfill[cur_loc[0], cur_loc[1] - 1] <= gameboard_floodfill[
                cur_loc[0], cur_loc[1]]:
                count_denom += 1 - epsilon

            if cur_loc[0] - 1 >= 0 and gameboard_floodfill[cur_loc[0] - 1, cur_loc[1]] > gameboard_floodfill[
                cur_loc[0], cur_loc[1]]:
                count_denom += epsilon
            if cur_loc[0] + 1 < gameboard_floodfill.shape[0] and gameboard_floodfill[cur_loc[0] + 1, cur_loc[1]] > \
                    gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += epsilon
            if cur_loc[1] + 1 < gameboard_floodfill.shape[1] and gameboard_floodfill[cur_loc[0], cur_loc[1] + 1] > \
                    gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += epsilon
            if cur_loc[1] - 1 >= 0 and gameboard_floodfill[cur_loc[0], cur_loc[1] - 1] > gameboard_floodfill[
                cur_loc[0], cur_loc[1]]:
                count_denom += epsilon
            proba /= count_denom

            if self.reversed_complete_room_dict[get_nearest_regular(self.reversed_complete_room_dict, cur_loc)] == \
                    self.reversed_complete_room_dict[self.id_to_goal_tuple[candidate_idx]]:
                proba = 1

            self.previous_likelihood_dict[candidate_idx].append(proba)
            if len(self.previous_likelihood_dict[candidate_idx]) > 20:
                self.previous_likelihood_dict[candidate_idx].pop(0)

            likelihood_product = np.prod(self.previous_likelihood_dict[candidate_idx])

            if candidate_idx == victim_idx:
                # previous_likelihood_dict[candidate_idx] = likelihood_product * prob_goal
                victim_likelihood_product = likelihood_product * prob_goal
                denominator += likelihood_product * prob_goal

            else:
                # previous_likelihood_dict[candidate_idx] = likelihood_product * prob_non_goal
                denominator += likelihood_product * prob_non_goal

        # check for problem
        # if victim_likelihood_product == 0:
        #     print('problem at vic index: ', victim_idx)

        # Now compute posterior : P(traj|goal) * P(goal) / P(traj)
        # denom = sum(previous_likelihood_dict.values())
        # print('denom', denom)
        likelihood = victim_likelihood_product / denominator
        # print("likelihood = ", likelihood)
        if len(past_trajectory) < 15:
            return 1
        # most_likely_vic_idx = 0
        # most_likely_vic_posterior = 0
        # for candidate_idx in self.gameboard_floodfill_dictionary.keys():
        #     check_cand = np.prod(self.previous_likelihood_dict[candidate_idx]) / denominator
        #     if check_cand > most_likely_vic_posterior:
        #         most_likely_vic_posterior = check_cand
        #         most_likely_vic_idx = candidate_idx
        #
        # print()
        # print('most_likely_vic_idx: ', most_likely_vic_idx)
        # print('actual vic: ', victim_idx)

        return likelihood

    def process_past_traj(self, past_traj, just_saved=False):
        if just_saved:
            self.prev_past_traj = []
        for elem in past_traj:
            elem = eval(elem)
            self.prev_past_traj.append((elem[0] - 1, elem[1] - 1))
        return

    def process_past_traj_full(self, past_traj):
        past_traj_output = []
        for elem in past_traj:
            elem = eval(elem)
            past_traj_output.append((elem[0] - 1, elem[1] - 1))
        return past_traj_output

    def compute_new_level(self, start_location, end_location, next_victim_to_save_idx, past_traj, previously_saved):
        likelihood = self.check_player_floodfill_bayesian_efficient(past_traj, int(start_location[0]),
                                                                    int(start_location[1]), next_victim_to_save_idx,
                                                                    previously_saved)
        # print("visited_rooms_list", self.visited_rooms_list)
        # print('self.get_room(end_location[0], end_location[1])', self.check_room(end_location[0], end_location[1]))
        # print()
        # print('likelihood', likelihood)
        if likelihood < 0.3:
            return 1
        if likelihood > 0.99:
            # if self.check_room(end_location[0], end_location[1]) in self.visited_rooms_list and self.check_room(end_location[0], end_location[1]) !='undefined':
            # if self.check_room(end_location[0], end_location[1]) in self.visited_rooms_list and self.check_room(start_location[0], start_location[1]) in self.visited_rooms_list:
            if self.check_room(end_location[0], end_location[1]) == self.check_room(start_location[0], start_location[1]):
                return 3

        return 2

    def generate_adaptive_instructions(self, start_location, end_location, curr_heading, victim_save_record,
                                                 past_traj_input, next_victim_to_save_id,game_time,
                                                 to_plot=False):

        past_traj = self.process_past_traj_full(past_traj_input['x'])
        self.level_2_hold_counter += 1
        # print("self.goal+tuple id", self.goal_tuple_to_id)
        # print("id_to_goal_tuple", self.id_to_goal_tuple)
        # print("past_traj", past_traj)
        previously_saved = []
        # print("goal_tuple_to_id : ",self.goal_tuple_to_id)
        for save_order in victim_save_record:
            x = victim_save_record[save_order]['x']
            y = victim_save_record[save_order]['y']
            vic_idx = self.goal_tuple_to_id[str((x, y))]
            previously_saved.append(vic_idx)

        # At 2 min, clear history
        if game_time <= 120 and game_time > 115:
            for candidate_idx in self.id_to_goal_tuple:
                if candidate_idx in previously_saved:
                    self.previous_likelihood_dict[candidate_idx] = [0] * 20
                else:
                    self.previous_likelihood_dict[candidate_idx] = [1] * 20

        if self.previously_saved != previously_saved:
            self.level = 2
            self.process_past_traj(past_traj_input['x'], just_saved=True)
            for candidate_idx in self.id_to_goal_tuple:
                self.previous_likelihood_dict[candidate_idx] = [1] * 20

        else:
            self.process_past_traj(past_traj_input['x'], just_saved=False)

        if self.level == 1:
            advice_output = self.generate_level_1_instructions(start_location, end_location,
                                                                         curr_heading,
                                                                         victim_save_record)
            output = advice_output

        elif self.level == 2:

            advice_output, final_dest = self.generate_level_2_instructions(start_location, end_location,
                                                                                     curr_heading, victim_save_record,
                                                                                     past_traj)
            if self.level_2_hold_counter >= 5:
                self.level_2_hold_counter = 0
                self.level_2_hold_advice = advice_output
                output = advice_output
            else:
                output = self.level_2_hold_advice

        else:
            if self.get_room(end_location[0], end_location[1]) in self.visited_rooms_list:
                advice_output = self.generate_level_3_instructions(start_location, end_location,
                                                                             curr_heading,
                                                                             victim_save_record)
            else:
                self.level = 2
                advice_output, final_dest = self.generate_level_2_instructions(start_location,
                                                                                         end_location,
                                                                                         curr_heading,
                                                                                         victim_save_record,
                                                                                         past_traj_input)

            output = advice_output

        if self.level == 1 and self.previously_saved == previously_saved:
            self.level = 1
        elif self.previously_saved != previously_saved or self.level_change_counter > 2:
            self.level_change_counter = 0
            try:
                self.new_level = self.compute_new_level(start_location, end_location, next_victim_to_save_id, past_traj,
                                                    previously_saved)
            except:
                self.new_level = self.level

            # if self.level == 2 and self.new_level == 3:
            #     self.level = 2
            self.level = self.new_level

        self.level_change_counter += 1
        self.previously_saved = previously_saved
        # print('LEVEL = ', self.level)
        return output










































