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

from mission.static.js.py_files.a_star import Search, Decision_Point_Search, Decision_Point_Search_Augmented_Map2
from mission.static.js.py_files.gameboard_utils import *
from mission.static.js.py_files.settings import *
from mission.static.js.py_files.advice_utils import recompute_path
import csv

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
    # print('obstacles', obstacles)
    # print('yellow_locs', yellow_locs)
    # print('green_locs',green_locs)
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

def read_csv():
    with open(prefix+'setup_files/minecraft_tilemap_2_second_map_with_victims.csv', newline='') as f:
        reader = csv.reader(f)
        data = list([[int(x) for x in reader_item] for reader_item in reader])
    doors = []
    data = np.array(data)
    xy_data = np.zeros((data.shape[1], data.shape[0]))
    obstacles_list = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j] == 1:
                obstacles_list.append((i,j))
            xy_data[j,i] = data[i,j]
            if data[i,j] == 2:
                doors.append((i,j))

    rc_data = data
    yellows = []
    greens = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j]==5:
                yellows.append((j, i))
            elif data[i, j]==6:
                greens.append((j, i))

    xy_yellows = yellows
    xy_greens = greens
    rc_yellows = [(j, i) for (i, j) in yellows]
    rc_greens = [(j, i) for (i, j) in greens]

    id_to_goal_tuple = {}
    counter = 0
    for i in range(len(rc_yellows)):
        id_to_goal_tuple[counter] = rc_yellows[i]
        counter += 1
    for i in range(len(rc_greens)):
        id_to_goal_tuple[counter] = rc_greens[i]
        counter += 1

    return rc_data, xy_data, xy_yellows, xy_greens, rc_yellows, rc_greens, obstacles_list, id_to_goal_tuple, doors

def load_goal_distances(cutoff_victims=25):
    with open(prefix+'setup_files/map2_goal_distances_2.pkl', 'rb') as file:
        goal_distances = pkl.load(file)
    goal_tuple_to_id = {}
    counter = 0

    goal_distances_keys = list(goal_distances.keys())
    for key in goal_distances_keys:
        key1 = eval(key.split(':')[0])
        key2 = eval(key.split(':')[1])
        key1 = str((key1[1], key1[0]))
        key2 = str((key2[1], key2[0]))

        if key1 not in goal_tuple_to_id and len(goal_tuple_to_id.keys()) < cutoff_victims:
            goal_tuple_to_id[key1] = counter
            counter += 1
        if key2 not in goal_tuple_to_id and len(goal_tuple_to_id.keys()) < cutoff_victims:
            goal_tuple_to_id[key2] = counter
            counter += 1

    num_goals = len(goal_tuple_to_id.keys())
    # print('num_goals', num_goals)

    goal_tuple_to_id_new = {}
    counter = 0
    for key, value in list(goal_tuple_to_id.items()):
        goal_tuple_to_id_new[key] = counter
        counter += 1
    goal_tuple_to_id = copy.deepcopy(goal_tuple_to_id_new)

    id_to_goal_tuple = {v: eval(k) for k, v in goal_tuple_to_id.items()}

    distance_matrix = np.zeros((num_goals, num_goals))
    for key in list(goal_distances.keys()):
        if key.split(':')[0] not in goal_tuple_to_id or key.split(':')[1] not in goal_tuple_to_id:
            continue
        key1 = goal_tuple_to_id[key.split(':')[0]]
        key2 = goal_tuple_to_id[key.split(':')[1]]
        d = goal_distances[key]
        distance_matrix[key1, key2] = d-2

    distance_dict = {}
    for key in list(goal_distances.keys()):
        if key.split(':')[0] not in goal_tuple_to_id or key.split(':')[1] not in goal_tuple_to_id:
            continue
        d = goal_distances[key]
        key1 = goal_tuple_to_id[key.split(':')[0]]
        key2 = goal_tuple_to_id[key.split(':')[1]]
        if key1 not in distance_dict:
            distance_dict[key1] = {}
        distance_dict[key1][key2] = d
    return goal_tuple_to_id, id_to_goal_tuple, distance_dict, distance_matrix


class Map_2_Instructor:
    def __init__(self):
        self.advice_counter = 0
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

        self.starting_bool = True
        self.goal_tuple_to_id, self.id_to_goal_tuple, self.distance_dict, self.distance_matrix = load_goal_distances()
        self.stairs = []
        self.inverse_gameboard, self.original_gameboard, self.yellow_locs, self.green_locs, self.flipped_yellow_locs, self.flipped_green_locs, self.obstacles, self.id_to_goal_tuple, self.doors = read_csv()

        for key in self.id_to_goal_tuple:
            self.id_to_goal_tuple[key] = (self.id_to_goal_tuple[key][1], self.id_to_goal_tuple[key][0])

        self.rooms_dictionary, self.position_to_room_dict, self.complete_room_dict, self.reversed_complete_room_dict = self.get_rooms_dict(
            self.inverse_gameboard)

        with open(prefix + 'setup_files/aggregate_path_4.pkl', 'rb') as file:
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

        for i in range(self.inverse_gameboard.shape[0]):
            for j in range(self.inverse_gameboard.shape[1]):
                if self.inverse_gameboard[i, j] == 1:
                    self.gameboard[i, j] = [237, 184, 121]
                if self.inverse_gameboard[i, j] == 2:
                    self.gameboard[i, j] = [185, 116, 85]

        for room_name in self.complete_room_dict:
            self.complete_room_dict[room_name] = [(j, i) for (i, j) in self.complete_room_dict[room_name]]

        self.flipped_obstacles = [(j, i) for (i, j) in self.obstacles]
        self.flipped_stairs = [(j, i) for (i, j) in self.stairs]
        self.flipped_doors = [(j, i) for (i, j) in self.doors]

        with open(prefix + 'setup_files/shortest_augmented_dps_distances_map2.pkl', 'rb') as handle:
            self.closest_decision_pt = pkl.load(handle)

        with open(prefix + "setup_files/door_index_to_dp_indices_dict_map2.pkl", 'rb') as handle:
            self.door_index_to_dp_indices_dict = pkl.load(handle)

        self.dp_dict = self.generate_decision_points_augmented()

        # with open(prefix + 'setup_files/dp_floodfill_dict_map2.pkl', 'rb') as filename:
        #     self.gameboard_floodfill_dictionary = pkl.load(filename)

        with open(prefix + 'setup_files/victim_floodfill_dict_map2_2.pkl', 'rb') as filename:
            self.gameboard_floodfill_dictionary = pkl.load(filename)

        self.previous_likelihood_dict = {}
        for candidate_idx in self.id_to_goal_tuple:
            self.previous_likelihood_dict[candidate_idx] = [1] * 20

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

    def run_floodfill_on_coordinate(self, coordinate, save=False, savefile=(prefix+'img/intersection_floodfill_.png')):
        x = coordinate[0]
        z = coordinate[1]
        # victim_idx = 6
        gameboard_floodfill = np.empty((self.gameboard.shape[0], self.gameboard.shape[1]))
        gameboard_floodfill.fill(100)
        print("gameboard_floodfill shape", gameboard_floodfill.shape)

        # stack = [id_to_goal_tuple[victim_idx]]
        # visited = []
        gameboard_floodfill[x,z] = 0

        OPEN_priQ = []
        heapq.heapify(OPEN_priQ)
        heapq.heappush(OPEN_priQ, (0, (x,z)))
        CLOSED = {}

        while not len(OPEN_priQ) == 0:
            dist, curr_location = heapq.heappop(OPEN_priQ)

            if curr_location in CLOSED:
                continue

            if (curr_location[0] + 1) < self.gameboard.shape[0]:
                # if (curr_location[0]+1, curr_location[1]) in CLOSED or (curr_location[0]+1, curr_location[1]) in obstacles:
                if (curr_location[0] + 1, curr_location[1]) not in CLOSED and (
                        curr_location[0] + 1, curr_location[1]) not in self.obstacles:
                    heapq.heappush(OPEN_priQ, (dist + 1, (curr_location[0] + 1, curr_location[1])))
                    gameboard_floodfill[curr_location[0] + 1, curr_location[1]] = dist + 1

            if (curr_location[0] - 1) >= 0:
                # if (curr_location[0] - 1, curr_location[1]) in CLOSED or (
                # curr_location[0] - 1, curr_location[1]) in obstacles:
                if (curr_location[0] - 1, curr_location[1]) not in CLOSED and (
                        curr_location[0] - 1, curr_location[1]) not in self.obstacles:
                    heapq.heappush(OPEN_priQ, (dist + 1, (curr_location[0] - 1, curr_location[1])))
                    gameboard_floodfill[curr_location[0] - 1, curr_location[1]] = dist + 1

            if (curr_location[1] + 1) < self.gameboard.shape[1]:
                if (curr_location[0], curr_location[1] + 1) not in CLOSED and (
                        curr_location[0], curr_location[1] + 1) not in self.obstacles:
                    # if (curr_location[0], curr_location[1]+1) in CLOSED or (
                    #         curr_location[0], curr_location[1]+1) in obstacles:
                    heapq.heappush(OPEN_priQ, (dist + 1, (curr_location[0], curr_location[1] + 1)))
                    gameboard_floodfill[curr_location[0], curr_location[1] + 1] = dist + 1

            if (curr_location[1] - 1) >= 0:
                if (curr_location[0], curr_location[1] - 1) not in CLOSED and (
                        curr_location[0], curr_location[1] - 1) not in self.obstacles:
                    # if (curr_location[0], curr_location[1] - 1) in CLOSED or (
                    #         curr_location[0], curr_location[1] - 1) in obstacles:

                    heapq.heappush(OPEN_priQ, (dist + 1, (curr_location[0], curr_location[1] - 1)))
                    gameboard_floodfill[curr_location[0], curr_location[1] - 1] = dist + 1
            CLOSED[curr_location] = True

        # gameboard_floodfill = np.reshape(gameboard_floodfill, (gameboard_floodfill.shape[0], gameboard_floodfill.shape[1], 3))
        # gameboard[id_to_goal_tuple[victim_idx][0], id_to_goal_tuple[victim_idx][1]] = [100000]
        if save:
            plt.imshow(gameboard_floodfill.astype(np.uint64))
            plt.savefig(savefile)
            plt.close()
        return gameboard_floodfill

    def run_floodfill_on_victims(self):
        # Run Floodfill on Victims
        victim_floodfill_dict = {}
        print("id_to_goal_tuple", self.id_to_goal_tuple)

        for idx in self.id_to_goal_tuple:
            victim_location = (self.id_to_goal_tuple[idx][1], self.id_to_goal_tuple[idx][0])
            print("victim_location", victim_location)
            victim_floodfill_dict[idx] = self.run_floodfill_on_coordinate(victim_location, save=False,
                                                                                   savefile=(prefix+'img/victim_map2_floodfill_'+str(idx)+'_.png'))


        with open(prefix+"setup_files/victim_floodfill_dict_map2_3.pkl", 'wb') as filename:
            pkl.dump(victim_floodfill_dict, filename)
        print("DONE")

    def generate_decision_points_augmented(self):
        all_decision_points_xy = {
            0: {
                "type": "start",
                "location": (26, 9),
                "neighbors": [1, 228],
                "within_room_name": "North Corridor",
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
                "type": "passing_door",
                "location": (28, 9),
                "neighbors": [2, 7, 0],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 2,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [2],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            2: {
                "type": "entrance_out",
                "location": (28, 8),
                "neighbors": [1, 3],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 2,
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
            3: {
                "type": "door",
                "location": (28, 7),
                "neighbors": [2, 4],
                "within_room_name": "Room 204",
                "color": "",
                "door_index": 2,
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
            4: {
                "type": "entrance_in",
                "location": (28, 6),
                "neighbors": [5, 3, 6],
                "within_room_name": "Room 204",
                "color": "",
                "door_index": 2,
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
            5: {
                "type": "room_center",
                "location": (25, 3),
                "neighbors": [6, 4],
                "within_room_name": "Room 204",
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
            6: {
                "type": "victim",
                "location": (23, 6),
                "neighbors": [5, 4],
                "within_room_name": "Room 204",
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
            7: {
                "type": "intersection, passing_door",
                "location": (37, 9),
                "neighbors": [8, 1, 12, 130],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 3,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [3],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            8: {
                "type": "entrance_out",
                "location": (37, 8),
                "neighbors": [7, 9],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 3,
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
            9: {
                "type": "door",
                "location": (37, 7),
                "neighbors": [10, 8],
                "within_room_name": "Room 205",
                "color": "",
                "door_index": 3,
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
            10: {
                "type": "entrance_in",
                "location": (37, 6),
                "neighbors": [11, 9],
                "within_room_name": "Room 205",
                "color": "",
                "door_index": 3,
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
            11: {
                "type": "room_center",
                "location": (34, 3),
                "neighbors": [10],
                "within_room_name": "Room 205",
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
            12: {
                "type": "passing_door",
                "location": (44, 9),
                "neighbors": [7, 13, 14, 21],
                "within_room_name": "North Corridor",
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
            13: {
                "type": "entrance_out",
                "location": (44, 8),
                "neighbors": [12, 15],
                "within_room_name": "North Corridor",
                "color": "",
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
            14: {
                "type": "entrance_out",
                "location": (45, 8),
                "neighbors": [12, 16],
                "within_room_name": "North Corridor",
                "color": "",
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
            15: {
                "type": "door",
                "location": (44, 7),
                "neighbors": [13, 17],
                "within_room_name": "Corporate Suite 1",
                "color": "",
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
            16: {
                "type": "door",
                "location": (45, 7),
                "neighbors": [14, 18],
                "within_room_name": "Corporate Suite 1",
                "color": "",
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
            17: {
                "type": "entrance_in",
                "location": (44, 6),
                "neighbors": [15, 19, 20],
                "within_room_name": "Corporate Suite 1",
                "color": "",
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
            18: {
                "type": "entrance_in",
                "location": (45, 6),
                "neighbors": [16, 19, 20],
                "within_room_name": "Corporate Suite 1",
                "color": "",
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
            19: {
                "type": "room_center",
                "location": (44, 3),
                "neighbors": [20, 17, 18],
                "within_room_name": "Corporate Suite 1",
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
            20: {
                "type": "victim",
                "location": (39, 1),
                "neighbors": [19, 17, 18],
                "within_room_name": "Corporate Suite 1",
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
            21: {
                "type": "passing_door",
                "location": (57, 9),
                "neighbors": [12, 30, 22, 23],
                "within_room_name": "North Corridor",
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
            22: {
                "type": "entrance_out",
                "location": (57, 8),
                "neighbors": [21, 24],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 5,
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
            23: {
                "type": "entrance_out",
                "location": (58, 8),
                "neighbors": [21, 25],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 5,
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
                "type": "door",
                "location": (57, 7),
                "neighbors": [22, 26],
                "within_room_name": "Corporate Suite 2",
                "color": "",
                "door_index": 5,
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
            25: {
                "type": "door",
                "location": (58, 7),
                "neighbors": [23, 27],
                "within_room_name": "Corporate Suite 2",
                "color": "",
                "door_index": 5,
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
            26: {
                "type": "entrance_in",
                "location": (57, 6),
                "neighbors": [28, 24, 29],
                "within_room_name": "Corporate Suite 2",
                "color": "",
                "door_index": 5,
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
            27: {
                "type": "entrance_in",
                "location": (58, 6),
                "neighbors": [28, 29, 25],
                "within_room_name": "Corporate Suite 2",
                "color": "",
                "door_index": 5,
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
            28: {
                "type": "room_center",
                "location": (57, 3),
                "neighbors": [29, 26, 27],
                "within_room_name": "Corporate Suite 2",
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
            29: {
                "type": "victim",
                "location": (63, 2),
                "neighbors": [28, 26, 27],
                "within_room_name": "Corporate Suite 2",
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
            30: {
                "type": "intersection",
                "location": (64, 9),
                "neighbors": [21, 31, 149],
                "within_room_name": "Zigzag Hallway",
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
            31: {
                "type": "passing_door",
                "location": (66, 9),
                "neighbors": [30, 32, 37],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 6,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [6],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            32: {
                "type": "entrance_out",
                "location": (66, 8),
                "neighbors": [31, 33],
                "within_room_name": "North Corridor",
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
            33: {
                "type": "door",
                "location": (66, 7),
                "neighbors": [32, 34],
                "within_room_name": "Room 206",
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
            34: {
                "type": "entrance_in",
                "location": (66, 6),
                "neighbors": [35, 36, 33],
                "within_room_name": "Room 206",
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
            35: {
                "type": "room_corner",
                "location": (66, 3),
                "neighbors": [34, 36],
                "within_room_name": "Room 206",
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
            36: {
                "type": "victim",
                "location": (70, 6),
                "neighbors": [35, 34],
                "within_room_name": "Room 206",
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
            37: {
                "type": "passing_door",
                "location": (75, 9),
                "neighbors": [31, 38, 42, 49, 50, 51],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 7,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [7, 11],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            38: {
                "type": "entrance_out",
                "location": (75, 8),
                "neighbors": [37, 39],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 7,
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
            39: {
                "type": "door",
                "location": (75, 7),
                "neighbors": [40, 38],
                "within_room_name": "Room 207",
                "color": "",
                "door_index": 7,
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
            40: {
                "type": "entrance_in",
                "location": (75, 6),
                "neighbors": [41, 39],
                "within_room_name": "Room 207",
                "color": "",
                "door_index": 7,
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
            41: {
                "type": "room_corner",
                "location": (75, 2),
                "neighbors": [40],
                "within_room_name": "Room 207",
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
            42: {
                "type": "passing_door",
                "location": (84, 9),
                "neighbors": [37, 48, 43],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 8,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [8],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            43: {
                "type": "entrance_out",
                "location": (84, 8),
                "neighbors": [42, 44],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 8,
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
            44: {
                "type": "door",
                "location": (84, 7),
                "neighbors": [43, 45],
                "within_room_name": "Room 208",
                "color": "",
                "door_index": 8,
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
                "type": "entrance_in",
                "location": (84, 6),
                "neighbors": [46, 47, 44],
                "within_room_name": "Room 208",
                "color": "",
                "door_index": 8,
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
                "type": "room_corner",
                "location": (84, 2),
                "neighbors": [45, 47],
                "within_room_name": "Room 208",
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
                "type": "victim",
                "location": (90, 6),
                "neighbors": [46, 45],
                "within_room_name": "Room 208",
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
            48: {
                "type": "intersection",
                "location": (86, 9),
                "neighbors": [42, 63],
                "within_room_name": "Right Hallway",
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
            49: {
                "type": "entrance_out",
                "location": (74, 11),
                "neighbors": [37, 52],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 11,
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
            50: {
                "type": "entrance_out",
                "location": (75, 11),
                "neighbors": [37, 53],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 11,
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
            51: {
                "type": "entrance_out",
                "location": (76, 11),
                "neighbors": [37, 54],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 11,
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
                "type": "door",
                "location": (74, 12),
                "neighbors": [49, 55],
                "within_room_name": "Taylor Auditorium",
                "color": "",
                "door_index": 11,
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
                "type": "door",
                "location": (75, 12),
                "neighbors": [50, 56],
                "within_room_name": "Taylor Auditorium",
                "color": "",
                "door_index": 11,
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
                "type": "door",
                "location": (76, 12),
                "neighbors": [51, 57],
                "within_room_name": "Taylor Auditorium",
                "color": "",
                "door_index": 11,
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
            55: {
                "type": "entrance_in",
                "location": (74, 13),
                "neighbors": [52, 61, 58],
                "within_room_name": "Taylor Auditorium",
                "color": "",
                "door_index": 11,
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
            56: {
                "type": "entrance_in",
                "location": (75, 13),
                "neighbors": [53, 61, 58],
                "within_room_name": "Taylor Auditorium",
                "color": "",
                "door_index": 11,
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
            57: {
                "type": "entrance_in",
                "location": (76, 13),
                "neighbors": [54, 61, 58],
                "within_room_name": "Taylor Auditorium",
                "color": "",
                "door_index": 11,
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
            58: {
                "type": "room_corner",
                "location": (81, 14),
                "neighbors": [55, 56, 57, 59, 61, 60],
                "within_room_name": "Taylor Auditorium",
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
            59: {
                "type": "room_corner",
                "location": (81, 24),
                "neighbors": [62, 60, 58, 61],
                "within_room_name": "Taylor Auditorium",
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
                "type": "room_corner",
                "location": (69, 24),
                "neighbors": [61, 59, 58, 62],
                "within_room_name": "Taylor Auditorium",
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
                "type": "room_corner",
                "location": (69, 14),
                "neighbors": [55, 56, 57, 58, 59, 62, 60],
                "within_room_name": "Taylor Auditorium",
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
            62: {
                "type": "victim",
                "location": (78, 21),
                "neighbors": [59, 60],
                "within_room_name": "Taylor Auditorium",
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
            63: {
                "type": "passing_door",
                "location": (86, 29),
                "neighbors": [64, 65, 48, 72],
                "within_room_name": "Right Hallway",
                "color": "",
                "door_index": 15,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [15],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            64: {
                "type": "entrance_out",
                "location": (84, 29),
                "neighbors": [66, 63],
                "within_room_name": "Right Hallway",
                "color": "",
                "door_index": 15,
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
            65: {
                "type": "entrance_out",
                "location": (84, 30),
                "neighbors": [67, 63],
                "within_room_name": "Right Hallway",
                "color": "",
                "door_index": 15,
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
            66: {
                "type": "door",
                "location": (83, 29),
                "neighbors": [68, 64],
                "within_room_name": "Taylor Sound Room",
                "color": "",
                "door_index": 15,
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
            67: {
                "type": "door",
                "location": (83, 30),
                "neighbors": [69, 65],
                "within_room_name": "Taylor Sound Room",
                "color": "",
                "door_index": 15,
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
            68: {
                "type": "entrance_in",
                "location": (82, 29),
                "neighbors": [66, 70],
                "within_room_name": "Taylor Sound Room",
                "color": "",
                "door_index": 15,
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
                "type": "entrance_in",
                "location": (82, 30),
                "neighbors": [67, 70],
                "within_room_name": "Taylor Sound Room",
                "color": "",
                "door_index": 15,
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
                "type": "room_center",
                "location": (76, 27),
                "neighbors": [71, 68, 69],
                "within_room_name": "Taylor Sound Room",
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
            71: {
                "type": "victim",
                "location": (71, 27),
                "neighbors": [70],
                "within_room_name": "Taylor Sound Room",
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
            72: {
                "type": "passing_door",
                "location": (86, 34),
                "neighbors": [73, 74, 63, 81],
                "within_room_name": "Right Hallway",
                "color": "",
                "door_index": 16,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [16],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            73: {
                "type": "entrance_out",
                "location": (84, 34),
                "neighbors": [75, 72],
                "within_room_name": "Right Hallway",
                "color": "",
                "door_index": 16,
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
                "type": "entrance_out",
                "location": (84, 35),
                "neighbors": [72, 76],
                "within_room_name": "Right Hallway",
                "color": "",
                "door_index": 16,
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
                "type": "door",
                "location": (83, 34),
                "neighbors": [77, 73],
                "within_room_name": "Creativity Nook",
                "color": "",
                "door_index": 16,
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
            76: {
                "type": "door",
                "location": (83, 35),
                "neighbors": [78, 74],
                "within_room_name": "Creativity Nook",
                "color": "",
                "door_index": 16,
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
            77: {
                "type": "entrance_in",
                "location": (82, 34),
                "neighbors": [79, 75],
                "within_room_name": "Creativity Nook",
                "color": "",
                "door_index": 16,
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
            78: {
                "type": "entrance_in",
                "location": (82, 35),
                "neighbors": [79, 76],
                "within_room_name": "Creativity Nook",
                "color": "",
                "door_index": 16,
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
            79: {
                "type": "room_corner",
                "location": (82, 42),
                "neighbors": [77, 78, 80],
                "within_room_name": "Creativity Nook",
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
            80: {
                "type": "victim",
                "location": (75, 34),
                "neighbors": [79],
                "within_room_name": "Creativity Nook",
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
            81: {
                "type": "intersection",
                "location": (86, 46),
                "neighbors": [72, 82],
                "within_room_name": "South Corridor",
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
            82: {
                "type": "intersection",
                "location": (56, 46),
                "neighbors": [81, 210, 83],
                "within_room_name": "South Corridor",
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
            83: {
                "type": "passing_door",
                "location": (56, 35),
                "neighbors": [82, 84, 85, 92],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 17,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [17],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            84: {
                "type": "entrance_out",
                "location": (58, 34),
                "neighbors": [83, 86],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 17,
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
                "type": "entrance_out",
                "location": (58, 35),
                "neighbors": [83, 87],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 17,
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
            86: {
                "type": "door",
                "location": (59, 34),
                "neighbors": [84, 88],
                "within_room_name": "Technology Nook",
                "color": "",
                "door_index": 17,
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
            87: {
                "type": "door",
                "location": (59, 35),
                "neighbors": [85, 89],
                "within_room_name": "Technology Nook",
                "color": "",
                "door_index": 17,
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
            88: {
                "type": "entrance_in",
                "location": (60, 34),
                "neighbors": [86, 90],
                "within_room_name": "Technology Nook",
                "color": "",
                "door_index": 17,
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
            89: {
                "type": "entrance_in",
                "location": (60, 35),
                "neighbors": [87, 90],
                "within_room_name": "Technology Nook",
                "color": "",
                "door_index": 17,
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
            90: {
                "type": "room_corner",
                "location": (60, 42),
                "neighbors": [88, 89, 91],
                "within_room_name": "Technology Nook",
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
            91: {
                "type": "victim",
                "location": (70, 35),
                "neighbors": [90],
                "within_room_name": "Technology Nook",
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
            92: {
                "type": "passing_door",
                "location": (56, 31),
                "neighbors": [93, 94, 143, 83],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 18,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [18],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            93: {
                "type": "entrance_out",
                "location": (54, 30),
                "neighbors": [92, 95],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 18,
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
                "type": "entrance_out",
                "location": (54, 31),
                "neighbors": [96, 92],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 18,
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
            95: {
                "type": "door",
                "location": (53, 30),
                "neighbors": [93, 97],
                "within_room_name": "Michelin Conference Room",
                "color": "",
                "door_index": 18,
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
                "location": (53, 31),
                "neighbors": [94, 98],
                "within_room_name": "Michelin Conference Room",
                "color": "",
                "door_index": 18,
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
            97: {
                "type": "entrance_in",
                "location": (52, 30),
                "neighbors": [95, 99, 103, 102, 101, 100],
                "within_room_name": "Michelin Conference Room",
                "color": "",
                "door_index": 18,
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
            98: {
                "type": "entrance_in",
                "location": (52, 31),
                "neighbors": [99, 96, 103, 102, 101, 100],
                "within_room_name": "Michelin Conference Room",
                "color": "",
                "door_index": 18,
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
                "type": "room_corner",
                "location": (52, 29),
                "neighbors": [97, 98, 102, 103, 101, 100],
                "within_room_name": "Michelin Conference Room",
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
            100: {
                "type": "room_corner",
                "location": (52, 37),
                "neighbors": [98, 97, 99, 103, 102, 101, 104, 105],
                "within_room_name": "Michelin Conference Room",
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
            101: {
                "type": "room_corner",
                "location": (41, 37),
                "neighbors": [98, 97, 99, 103, 102, 100, 104, 105],
                "within_room_name": "Michelin Conference Room",
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
            102: {
                "type": "room_corner",
                "location": (41, 30),
                "neighbors": [98, 97, 99, 103, 100, 101],
                "within_room_name": "Michelin Conference Room",
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
            103: {
                "type": "victim",
                "location": (45, 31),
                "neighbors": [102, 97, 98, 99, 100, 101],
                "within_room_name": "Michelin Conference Room",
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
            104: {
                "type": "entrance_out",
                "location": (46, 37),
                "neighbors": [106, 101, 100],
                "within_room_name": "Michelin Conference Room",
                "color": "",
                "door_index": 19,
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
                "type": "entrance_out",
                "location": (47, 37),
                "neighbors": [101, 100, 107],
                "within_room_name": "Michelin Conference Room",
                "color": "",
                "door_index": 19,
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
            106: {
                "type": "door",
                "location": (46, 38),
                "neighbors": [104, 108],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 19,
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
            107: {
                "type": "door",
                "location": (47, 38),
                "neighbors": [105, 109],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 19,
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
            108: {
                "type": "entrance_in",
                "location": (46, 39),
                "neighbors": [106, 110, 111],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 19,
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
                "type": "entrance_in",
                "location": (47, 39),
                "neighbors": [107, 110, 111],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 19,
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
            110: {
                "type": "entrance_in",
                "location": (41, 42),
                "neighbors": [112, 108, 109],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 20,
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
            111: {
                "type": "entrance_in",
                "location": (41, 43),
                "neighbors": [108, 109, 113],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 20,
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
            112: {
                "type": "door",
                "location": (40, 42),
                "neighbors": [114, 110],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 20,
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
            113: {
                "type": "door",
                "location": (40, 43),
                "neighbors": [115, 111],
                "within_room_name": "Backstage Room",
                "color": "",
                "door_index": 20,
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
                "type": "entrance_out",
                "location": (39, 42),
                "neighbors": [116, 112],
                "within_room_name": "Middle Hallway",
                "color": "",
                "door_index": 20,
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
                "type": "entrance_out",
                "location": (39, 43),
                "neighbors": [116, 113],
                "within_room_name": "Middle Hallway",
                "color": "",
                "door_index": 20,
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
            116: {
                "type": "passing_door",
                "location": (37, 42),
                "neighbors": [114, 115, 210, 117],
                "within_room_name": "Middle Hallway",
                "color": "",
                "door_index": 20,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [20],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            117: {
                "type": "passing_door",
                "location": (37, 24),
                "neighbors": [118, 119, 116, 130],
                "within_room_name": "Middle Hallway",
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
            118: {
                "type": "entrance_out",
                "location": (39, 24),
                "neighbors": [117, 120],
                "within_room_name": "Middle Hallway",
                "color": "",
                "door_index": 13,
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
            119: {
                "type": "entrance_out",
                "location": (39, 25),
                "neighbors": [117, 121],
                "within_room_name": "Middle Hallway",
                "color": "",
                "door_index": 13,
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
            120: {
                "type": "door",
                "location": (40, 24),
                "neighbors": [118, 122],
                "within_room_name": "Carnegie Conference Room",
                "color": "",
                "door_index": 13,
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
            121: {
                "type": "door",
                "location": (40, 25),
                "neighbors": [119, 123],
                "within_room_name": "Carnegie Conference Room",
                "color": "",
                "door_index": 13,
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
            122: {
                "type": "entrance_in",
                "location": (41, 24),
                "neighbors": [120, 124, 125, 128, 127],
                "within_room_name": "Carnegie Conference Room",
                "color": "",
                "door_index": 13,
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
            123: {
                "type": "entrance_in",
                "location": (41, 25),
                "neighbors": [125, 121, 124, 125, 126, 127, 128],
                "within_room_name": "Carnegie Conference Room",
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
            124: {
                "type": "room_corner",
                "location": (42, 25),
                "neighbors": [122, 123, 125, 128, 127],
                "within_room_name": "Carnegie Conference Room",
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
            125: {
                "type": "room_corner",
                "location": (42, 14),
                "neighbors": [122, 123, 124, 125, 128, 126],
                "within_room_name": "Carnegie Conference Room",
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
            126: {
                "type": "room_corner",
                "location": (51, 14),
                "neighbors": [129, 125, 127],
                "within_room_name": "Carnegie Conference Room",
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
            127: {
                "type": "room_corner",
                "location": (51, 25),
                "neighbors": [124, 126],
                "within_room_name": "Carnegie Conference Room",
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
            128: {
                "type": "victim",
                "location": (48, 19),
                "neighbors": [125, 124],
                "within_room_name": "Carnegie Conference Room",
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
            129: {
                "type": "victim",
                "location": (52, 13),
                "neighbors": [126],
                "within_room_name": "Carnegie Conference Room",
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
            130: {
                "type": "passing_door",
                "location": (37, 14),
                "neighbors": [131, 132, 7, 117],
                "within_room_name": "Middle Hallway",
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
            131: {
                "type": "entrance_out",
                "location": (35, 14),
                "neighbors": [133, 130],
                "within_room_name": "Middle Hallway",
                "color": "",
                "door_index": 9,
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
            132: {
                "type": "entrance_out",
                "location": (35, 15),
                "neighbors": [134, 130],
                "within_room_name": "Middle Hallway",
                "color": "",
                "door_index": 9,
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
            133: {
                "type": "door",
                "location": (34, 14),
                "neighbors": [135, 131],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 9,
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
            134: {
                "type": "door",
                "location": (34, 15),
                "neighbors": [136, 132],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 9,
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
                "type": "entrance_in",
                "location": (33, 14),
                "neighbors": [137, 133, 140, 138, 141, 142],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 9,
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
                "type": "entrance_in",
                "location": (33, 15),
                "neighbors": [137, 134, 140, 138, 141, 142],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 9,
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
            137: {
                "type": "room_corner",
                "location": (32, 14),
                "neighbors": [135, 136, 140, 138, 141, 142],
                "within_room_name": "The Chemistry Lab",
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
            138: {
                "type": "room_corner",
                "location": (32, 32),
                "neighbors": [137, 135, 136, 142, 141, 139, 140, 160, 161, 173, 174],
                "within_room_name": "The Chemistry Lab",
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
            139: {
                "type": "room_corner",
                "location": (18, 32),
                "neighbors": [142, 141, 138, 140, 186, 187, 160, 161, 173, 174],
                "within_room_name": "The Chemistry Lab",
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
            140: {
                "type": "room_corner",
                "location": (18, 14),
                "neighbors": [141, 137, 139, 186, 187],
                "within_room_name": "The Chemistry Lab",
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
            141: {
                "type": "victim",
                "location": (23, 18),
                "neighbors": [140, 139, 137, 138, 160, 161, 173, 174],
                "within_room_name": "The Chemistry Lab",
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
            142: {
                "type": "victim",
                "location": (30, 30),
                "neighbors": [138, 137, 135, 136, 160, 161, 173, 174],
                "within_room_name": "The Chemistry Lab",
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
                "type": "intersection",
                "location": (56, 25),
                "neighbors": [92, 144],
                "within_room_name": "Zigzag Hallway",
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
            144: {
                "type": "passing_door",
                "location": (62, 25),
                "neighbors": [143, 148, 145],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 14,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [14],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            145: {
                "type": "entrance_out",
                "location": (62, 26),
                "neighbors": [144, 146],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 14,
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
            146: {
                "type": "door",
                "location": (62, 27),
                "neighbors": [145, 147],
                "within_room_name": "Room 120",
                "color": "",
                "door_index": 14,
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
            147: {
                "type": "entrance_in",
                "location": (62, 28),
                "neighbors": [146],
                "within_room_name": "Room 210",
                "color": "",
                "door_index": 14,
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
            148: {
                "type": "intersection",
                "location": (64, 25),
                "neighbors": [144, 149],
                "within_room_name": "Zigzag Hallway",
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
            149: {
                "type": "passing_door",
                "location": (64, 16),
                "neighbors": [30, 148, 150, 151, 152],
                "within_room_name": "Zigzag Hallway",
                "color": "",
                "door_index": 10,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [10],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            150: {
                "type": "entrance_out",
                "location": (63, 14),
                "neighbors": [153, 149],
                "within_room_name": "Zigzag Hallway",
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
            151: {
                "type": "entrance_out",
                "location": (63, 15),
                "neighbors": [154, 149],
                "within_room_name": "Zigzag Hallway",
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
            152: {
                "type": "entrance_out",
                "location": (63, 16),
                "neighbors": [155, 149],
                "within_room_name": "Zigzag Hallway",
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
            153: {
                "type": "door",
                "location": (62, 14),
                "neighbors": [156, 150],
                "within_room_name": "Room 209",
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
            154: {
                "type": "door",
                "location": (62, 15),
                "neighbors": [157, 151],
                "within_room_name": "Room 209",
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
            155: {
                "type": "door",
                "location": (62, 16),
                "neighbors": [158, 152],
                "within_room_name": "Room 209",
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
            156: {
                "type": "entrance_in",
                "location": (61, 14),
                "neighbors": [153, 159],
                "within_room_name": "Room 209",
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
            157: {
                "type": "entrance_in",
                "location": (61, 15),
                "neighbors": [154, 159],
                "within_room_name": "Room 209",
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
            158: {
                "type": "entrance_in",
                "location": (61, 16),
                "neighbors": [155, 159],
                "within_room_name": "Room 209",
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
            159: {
                "type": "victim",
                "location": (55, 21),
                "neighbors": [156, 157, 158],
                "within_room_name": "Room 209",
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
            160: {
                "type": "entrance_out",
                "location": (23, 33),
                "neighbors": [139, 138, 141, 142, 162],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 23,
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
            161: {
                "type": "entrance_out",
                "location": (24, 33),
                "neighbors": [139, 138, 141, 142, 163],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 23,
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
                "type": "door",
                "location": (23, 34),
                "neighbors": [160, 164],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 23,
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
                "type": "door",
                "location": (24, 34),
                "neighbors": [161, 165],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 23,
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
                "type": "entrance_in",
                "location": (23, 35),
                "neighbors": [162, 166, 169],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 23,
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
                "type": "entrance_in",
                "location": (24, 35),
                "neighbors": [163, 166, 169],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 23,
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
                "type": "entrance_out",
                "location": (19, 38),
                "neighbors": [164, 165, 167],
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
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            167: {
                "type": "door",
                "location": (19, 39),
                "neighbors": [166, 168],
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
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            168: {
                "type": "entrance_in",
                "location": (19, 40),
                "neighbors": [167],
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
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            169: {
                "type": "entrance_out",
                "location": (24, 38),
                "neighbors": [164, 165, 170],
                "within_room_name": "Women\'s Room",
                "color": "",
                "door_index": 25,
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
            170: {
                "type": "door",
                "location": (24, 39),
                "neighbors": [169, 171],
                "within_room_name": "Women\'s Stall 2",
                "color": "",
                "door_index": 25,
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
            171: {
                "type": "entrance_in",
                "location": (24, 40),
                "neighbors": [170, 172],
                "within_room_name": "Women\'s Stall 2",
                "color": "",
                "door_index": 25,
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
            172: {
                "type": "victim",
                "location": (22, 40),
                "neighbors": [171],
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
            173: {
                "type": "entrance_out",
                "location": (26, 33),
                "neighbors": [139, 138, 141, 142, 175],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 22,
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
                "type": "entrance_out",
                "location": (27, 33),
                "neighbors": [139, 138, 141, 142, 176],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 22,
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
            175: {
                "type": "door",
                "location": (26, 34),
                "neighbors": [173, 177],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 22,
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
            176: {
                "type": "door",
                "location": (27, 34),
                "neighbors": [174, 178],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 22,
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
            177: {
                "type": "entrance_in",
                "location": (26, 35),
                "neighbors": [175, 179, 182],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 22,
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
            178: {
                "type": "entrance_in",
                "location": (27, 35),
                "neighbors": [176, 179, 182],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 22,
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
            179: {
                "type": "entrance_out",
                "location": (26, 38),
                "neighbors": [177, 178, 180],
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
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            180: {
                "type": "door",
                "location": (26, 39),
                "neighbors": [179, 181],
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
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            181: {
                "type": "entrance_in",
                "location": (26, 40),
                "neighbors": [180],
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
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            182: {
                "type": "entrance_out",
                "location": (31, 38),
                "neighbors": [177, 178, 183],
                "within_room_name": "Men\'s Room",
                "color": "",
                "door_index": 21,
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
            183: {
                "type": "door",
                "location": (31, 39),
                "neighbors": [182, 184],
                "within_room_name": "Men\'s Stall 2",
                "color": "",
                "door_index": 21,
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
                "type": "entrance_in",
                "location": (31, 40),
                "neighbors": [183, 185],
                "within_room_name": "Men\'s Stall 2",
                "color": "",
                "door_index": 21,
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
                "type": "victim",
                "location": (31, 43),
                "neighbors": [184],
                "within_room_name": "Men\'s Stall 2",
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
            186: {
                "type": "entrance_in",
                "location": (17, 29),
                "neighbors": [140, 139, 188],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 29,
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
            187: {
                "type": "entrance_in",
                "location": (17, 30),
                "neighbors": [140, 139, 189],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 29,
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
            188: {
                "type": "door",
                "location": (16, 29),
                "neighbors": [186, 190],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 29,
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
                "type": "door",
                "location": (16, 30),
                "neighbors": [191, 187],
                "within_room_name": "The Chemistry Lab",
                "color": "",
                "door_index": 29,
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
                "type": "entrance_out",
                "location": (15, 29),
                "neighbors": [192, 188],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 29,
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
            191: {
                "type": "entrance_out",
                "location": (15, 30),
                "neighbors": [192, 189],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 29,
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
            192: {
                "type": "passing_door",
                "location": (13, 29),
                "neighbors": [190, 191, 193, 211],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 29,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [29],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            193: {
                "type": "passing_door",
                "location": (13, 32),
                "neighbors": [192, 200, 194, 195],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 30,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [30],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            194: {
                "type": "entrance_out",
                "location": (11, 32),
                "neighbors": [196, 193],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 30,
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
                "type": "entrance_out",
                "location": (11, 33),
                "neighbors": [193, 197],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 30,
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
                "type": "door",
                "location": (10, 32),
                "neighbors": [198, 194],
                "within_room_name": "Room 201",
                "color": "",
                "door_index": 30,
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
            197: {
                "type": "door",
                "location": (10, 33),
                "neighbors": [199, 195],
                "within_room_name": "Room 201",
                "color": "",
                "door_index": 30,
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
            198: {
                "type": "entrance_in",
                "location": (9, 32),
                "neighbors": [196],
                "within_room_name": "Room 201",
                "color": "",
                "door_index": 30,
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
            199: {
                "type": "entrance_in",
                "location": (9, 33),
                "neighbors": [197],
                "within_room_name": "Room 201",
                "color": "",
                "door_index": 30,
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
            200: {
                "type": "passing_door",
                "location": (13, 42),
                "neighbors": [201, 202, 209, 193],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 28,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [28],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            201: {
                "type": "entrance_out",
                "location": (11, 42),
                "neighbors": [203, 200],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 28,
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
                "type": "entrance_out",
                "location": (11, 43),
                "neighbors": [204, 200],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 28,
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
            203: {
                "type": "door",
                "location": (10, 42),
                "neighbors": [205, 201],
                "within_room_name": "Room 200",
                "color": "",
                "door_index": 28,
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
            204: {
                "type": "door",
                "location": (10, 43),
                "neighbors": [206, 202],
                "within_room_name": "Room 200",
                "color": "",
                "door_index": 28,
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
            205: {
                "type": "entrance_in",
                "location": (9, 42),
                "neighbors": [208, 207, 203],
                "within_room_name": "Room 200",
                "color": "",
                "door_index": 28,
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
            206: {
                "type": "entrance_in",
                "location": (9, 43),
                "neighbors": [208, 207, 204],
                "within_room_name": "Room 200",
                "color": "",
                "door_index": 28,
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
                "type": "room_corner",
                "location": (9, 47),
                "neighbors": [208, 205, 206],
                "within_room_name": "Room 200",
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
            208: {
                "type": "victim",
                "location": (5, 45),
                "neighbors": [207, 205, 206],
                "within_room_name": "Room 200",
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
            209: {
                "type": "intersection",
                "location": (13, 46),
                "neighbors": [200, 210],
                "within_room_name": "South Corridor",
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
            210: {
                "type": "intersection",
                "location": (37, 46),
                "neighbors": [209, 116, 82],
                "within_room_name": "South Corridor",
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
            211: {
                "type": "passing_door",
                "location": (13, 20),
                "neighbors": [192, 220, 212, 213],
                "within_room_name": "Left Hallway",
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
            212: {
                "type": "entrance_out",
                "location": (11, 20),
                "neighbors": [214, 211],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 12,
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
            213: {
                "type": "entrance_out",
                "location": (11, 21),
                "neighbors": [215, 211],
                "within_room_name": "Left Hallway",
                "color": "",
                "door_index": 12,
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
            214: {
                "type": "door",
                "location": (10, 20),
                "neighbors": [216, 212],
                "within_room_name": "Room 202",
                "color": "",
                "door_index": 12,
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
            215: {
                "type": "door",
                "location": (10, 21),
                "neighbors": [217, 213],
                "within_room_name": "Room 202",
                "color": "",
                "door_index": 12,
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
            216: {
                "type": "entrance_in",
                "location": (9, 20),
                "neighbors": [219, 218, 214],
                "within_room_name": "Room 202",
                "color": "",
                "door_index": 12,
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
            217: {
                "type": "entrance_in",
                "location": (9, 21),
                "neighbors": [218, 219, 215],
                "within_room_name": "Room 202",
                "color": "",
                "door_index": 12,
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
            218: {
                "type": "room_corner",
                "location": (9, 27),
                "neighbors": [219, 217, 216],
                "within_room_name": "Room 202",
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
            219: {
                "type": "victim",
                "location": (3, 27),
                "neighbors": [218, 216, 217],
                "within_room_name": "Room 202",
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
            220: {
                "type": "passing_door",
                "location": (12, 11),
                "neighbors": [227, 211, 221],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 0,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [0],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            221: {
                "type": "entrance_out",
                "location": (11, 11),
                "neighbors": [220, 222],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 0,
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
            222: {
                "type": "door",
                "location": (10, 11),
                "neighbors": [221, 223],
                "within_room_name": "President Kay\'s Office",
                "color": "",
                "door_index": 0,
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
            223: {
                "type": "entrance_in",
                "location": (9, 11),
                "neighbors": [222, 226, 224, 225],
                "within_room_name": "President Kay\'s Office",
                "color": "",
                "door_index": 0,
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
            224: {
                "type": "room_center",
                "location": (4, 11),
                "neighbors": [223, 226, 225],
                "within_room_name": "President Kay\'s Office",
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
            225: {
                "type": "victim",
                "location": (1, 17),
                "neighbors": [224, 226, 223],
                "within_room_name": "President Kay\'s Office",
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
            226: {
                "type": "victim",
                "location": (4, 5),
                "neighbors": [224, 225, 223],
                "within_room_name": "President Kay\'s Office",
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
            227: {
                "type": "intersection",
                "location": (14, 10),
                "neighbors": [220, 228],
                "within_room_name": "North Corridor",
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
            228: {
                "type": "passing_door",
                "location": (18, 9),
                "neighbors": [227, 1],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 1,
                "door_coordinates_list": [],
                "door_dp_indices_list": [],
                "entrance_in_coordinates_list": [],
                "entrance_out_coordinates_list": [],
                "entrance_in_dp_indices_list": [],
                "entrance_out_dp_indices_list": [],
                "passing_associated_door_indices": [1],
                "exiting_room_name": "",
                "entering_room_name": "",
            },
            229: {
                "type": "entrance_out",
                "location": (18, 8),
                "neighbors": [228, 230],
                "within_room_name": "North Corridor",
                "color": "",
                "door_index": 1,
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
            230: {
                "type": "door",
                "location": (18, 7),
                "neighbors": [229, 231],
                "within_room_name": "Room 203",
                "color": "",
                "door_index": 1,
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
            231: {
                "type": "entrance_in",
                "location": (18, 6),
                "neighbors": [230, 233, 232],
                "within_room_name": "Room 203",
                "color": "",
                "door_index": 1,
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
            232: {
                "type": "room_center",
                "location": (15, 3),
                "neighbors": [233, 231],
                "within_room_name": "Room 203",
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
            233: {
                "type": "victim",
                "location": (17, 1),
                "neighbors": [232, 231],
                "within_room_name": "Room 203",
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

        }
        return all_decision_points_xy

    def get_rooms_dict(self, gameboard):
        rooms_dictionary = {
            'Room 200': {'x': [0, 10], 'z': [40, 49]},
            'Room 201': {'x': [0, 10], 'z': [30, 40]},
            'Room 202': {'x': [0, 10], 'z': [18, 29]},
            'President Kay\'s Office': {'x': [0, 10], 'z': [0, 17]},
            'Room 203': {'x': [11, 19], 'z': [0, 7]},
            'Room 204': {'x': [20, 29], 'z': [0, 7]},
            'Room 205': {'x': [30, 38], 'z': [0, 7]},
            'Corporate Suite 1': {'x': [39, 51], 'z': [0, 7]},
            'Corporate Suite 2': {'x': [52, 65], 'z': [0, 7]},
            'Room 206': {'x': [66, 74], 'z': [0, 7]},
            'Room 207': {'x': [75, 83], 'z': [0, 7]},
            'Room 208': {'x': [84, 92], 'z': [0, 7]},
            'North Corridor': {'x': [11, 92], 'z': [8, 11]},
            'Left Hallway': {'x': [11, 15], 'z': [12, 49]},
            'Middle Hallway': {'x': [35, 39], 'z': [12, 44]},
            'Zigzag Hallway, 1': {'x': [54, 58], 'z': [27, 44]},
            'Zigzag Hallway, 2': {'x': [54, 66], 'z': [23, 26]},
            'Zigzag Hallway, 3': {'x': [63, 66], 'z': [12, 22]},
            'Right Hallway': {'x': [84, 92], 'z': [12, 44]},
            'South Corridor': {'x': [16, 92], 'z': [45, 49]},
            'The Chemistry Lab': {'x': [16, 34], 'z': [12, 33]},
            'Women\'s Room': {'x': [16, 25], 'z': [34, 38]},
            'Women\'s Stall 1': {'x': [16, 20], 'z': [38, 44]},
            'Women\'s Stall 2': {'x': [21, 25], 'z': [39, 44]},
            'Men\'s Room': {'x': [26, 34], 'z': [34, 38]},
            'Men\'s Stall 1': {'x': [26, 29], 'z': [39, 44]},
            'Men\'s Stall 2': {'x': [30, 34], 'z': [39, 44]},
            'Carnegie Conference Room': {'x': [40, 53], 'z': [12, 27]},
            'Michelin Conference Room': {'x': [40, 53], 'z': [28, 38]},
            'Backstage Room': {'x': [40, 53], 'z': [39, 44]},
            'Room 209': {'x': [54, 62], 'z': [12, 22]},
            'Taylor Auditorium, 1': {'x': [67, 83], 'z': [12, 24]},
            'Taylor Auditorium, 2': {'x': [67, 71], 'z': [25, 26]},
            'Taylor Auditorium, 3': {'x': [81, 83], 'z': [25, 26]},
            'Room 210': {'x': [59, 67], 'z': [27, 33]},
            'Taylor Sound Room, 1': {'x': [68, 83], 'z': [27, 33]},
            'Taylor Sound Room, 2': {'x': [72, 79], 'z': [25, 26]},
            'Technology Nook': {'x': [59, 71], 'z': [34, 44]},
            'Creativity Nook': {'x': [72, 83], 'z': [34, 44]},
        }

        position_to_room_dict = {}
        for i in range(gameboard.shape[0]):
            for j in range(gameboard.shape[1]):
                newkey = str(i) + ',' + str(j)
                selected_room = None
                for room in rooms_dictionary:
                    if i >= rooms_dictionary[room]['x'][0] and i <= rooms_dictionary[room]['x'][1]:
                        if j >= rooms_dictionary[room]['z'][0] and j <= rooms_dictionary[room]['z'][1]:
                            selected_room = room
                            break
                if selected_room is None:
                    selected_room = 'unidentified'
                position_to_room_dict[newkey] = selected_room

        complete_room_dict = {}
        for room in rooms_dictionary:
            complete_room_dict[room] = []
            for i in range(rooms_dictionary[room]['x'][0], rooms_dictionary[room]['x'][1] + 1):
                for j in range(rooms_dictionary[room]['z'][0], rooms_dictionary[room]['z'][1] + 1):
                    complete_room_dict[room].append((i, j))

        reversed_complete_room_dict = {}
        for key in complete_room_dict:
            for loc in complete_room_dict[key]:
                reversed_complete_room_dict[loc] = key
                if "Taylor Auditorium" in key:
                    reversed_complete_room_dict[loc] = "Taylor Auditorium"
                if "Zigzag Hallway" in key:
                    reversed_complete_room_dict[loc] = "Zigzag Hallway"
                if "Taylor Sound Room" in key:
                    reversed_complete_room_dict[loc] = "Taylor Sound Room"
        return rooms_dictionary, position_to_room_dict, complete_room_dict, reversed_complete_room_dict

    def get_room(self, pos_x, pos_y):
        if (pos_x, pos_y) in self.reversed_complete_room_dict:
            room_to_go = self.reversed_complete_room_dict[(pos_x, pos_y)]
            if room_to_go not in self.visited_rooms_list:
                self.visited_rooms_list.append(room_to_go)
        # elif (pos_x+1, pos_y+1) in self.reversed_complete_room_dict:
        #     room_to_go = self.reversed_complete_room_dict[(pos_x, pos_y)]
        # else:
        #     room_to_go = ''
        else:
            room_to_go = "undefined"
        return room_to_go

    def generate_victim_path_details(self):
        self.num_saved = 0
        self.victim_path_details = {
            0: {
                "id": 11,
                "location": (23, 6),
                'color': "green",
                'saved_state': False,
            },
            1: {
                "id": 0,
                "location": (4, 5),
                'color': "yellow",
                'saved_state': False,

            },
            2: {
                "id": 13,
                "location": (1, 17),
                'color': "green",
                'saved_state': False,
            },
            3: {
                "id": 17,
                "location": (3, 27),
                'color': "green",
                'saved_state': False,
            },
            4: {
                "id": 7,
                "location": (5, 45),
                'color': "yellow",
                'saved_state': False,
            },
            5: {
                "id": 21,
                "location": (22, 40),
                'color': "green",
                'saved_state': False,
            },
            6: {
                "id": 18,
                "location": (30, 30),
                'color': "green",
                'saved_state': False,
            },
            7: {
                "id": 6,
                "location": (31, 43),
                'color': "yellow",
                'saved_state': False,
            },
            8: {
                "id": 3,
                "location": (23, 18),
                'color': "yellow",
                'saved_state': False,
            },
            9: {
                "id": 14,
                "location": (48, 19),
                'color': "green",
                'saved_state': False,
            },
            10: {
                "id": 2,
                "location": (52, 13),
                'color': "yellow",
                'saved_state': False,
            },
            11: {
                "id": 9,
                "location": (39, 1),
                'color': "green",
                'saved_state': False,
            },
            12: {
                "id": 16,
                "location": (78, 21),
                'color': "green",
                'saved_state': False,
            },
            13: {
                "id": 1,
                "location": (90, 6),
                'color': "yellow",
                'saved_state': False,
            },
            14: {
                "id": 4,
                "location": (71, 27),
                'color': "yellow",
                'saved_state': False,
            },
            15: {
                "id": 5,
                "location": (75, 34),
                'color': "yellow",
                'saved_state': False,
            },
            16: {
                "id": 20,
                "location": (70, 35),
                'color': "green",
                'saved_state': False,
            },
            17: {
                "id": 19,
                "location": (45, 31),
                'color': "green",
                'saved_state': False,
            },
            18: {
                "id": 15,
                "location": (55, 21),
                'color': "green",
                'saved_state': False,
            },
            19: {
                "id": 12,
                "location": (70, 6),
                'color': "green",
                'saved_state': False,
            },
            20: {
                "id": 10,
                "location": (63, 2),
                'color': "green",
                'saved_state': False,
            },
            21: {
                "id": 8,
                "location": (17, 1),
                'color': "green",
                'saved_state': False,
            },

        }
        self.location_to_victim_path_idx = {}
        for index in self.victim_path_details:
            self.location_to_victim_path_idx[self.victim_path_details[index]['location']] = index

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
                                         self.original_gameboard, self.flipped_obstacles, self.yellow_locs,
                                         self.green_locs)
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
        if len(victim_save_record) > 0:
            max_key = max(list(victim_save_record.keys()))
            actual_start_location = (victim_save_record[max_key]['x'], victim_save_record[max_key]['y'])
        else:
            actual_start_location = (28,11)

        try:
            xy_path = recompute_path(start_location, end_location, self.id_to_goal_tuple,
                                     self.original_gameboard, self.flipped_obstacles, self.yellow_locs,
                                     self.green_locs)
        except Timeout.Timeout:
            print("PATH FAILED TO COMPUTE")
            xy_path = []
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

        advice = " \n".join(advice_list[:min(2, len(advice_list))])
        # advice = " ".join(advice_list)
        return advice

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

        # self.dp_dict = self.generate_decision_points_augmented()

        if start_location not in self.closest_decision_pt:
            return '', None
        start_idx = self.closest_decision_pt[start_location]

        goal_idx = self.closest_decision_pt[end_location]
        # if goal_idx == 43:
        #     goal_idx = 45

        decision_search = Decision_Point_Search_Augmented_Map2(self.dp_dict)
        # print("Looking for (start_idx, goal_idx): ", (start_idx, goal_idx))
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
        if travel_path[0] == 0:
            types_based_advice_list.append(" ")

        if len(travel_path) == 1:
            if self.dp_dict[travel_path[0]]['color'] == 'green':
                types_based_advice_list.append('Find and save BLUE victim in current room.')
            else:
                types_based_advice_list.append('Find and save RED victim in current room.')

        for i in range(len(types_list) - 1):
            #
            # print(f'on travel path i = {i}, {travel_path[i]}, full path = {travel_path}\n')
            if i == 0:
                next_direction = curr_heading
                # next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                #                                                 self.dp_dict[travel_path[i + 1]]['location'])
                # turn_direction = self.get_turn_direction(path_heading, next_direction)
                # if next_direction != curr_heading:
                #     types_based_advice_list.append("First turn "+ self.turn_advice_dict[turn_direction])

                if 'passing_door' in types_list[i] or 'intersection' in types_list[i]:
                    next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                                    self.dp_dict[travel_path[i + 1]]['location'])
                    # print("directions to go ", path_heading, next_direction)
                    if next_direction != curr_heading:

                        turn_direction = self.get_turn_direction(path_heading, next_direction)
                        # print("turn_direction", turn_direction)
                        if turn_direction is not None:
                            if types_list[i + 1] == 'entrance_out':
                                types_based_advice_list.append(
                                    "Turn " + self.turn_advice_dict[turn_direction] + ' at door.')
                            else:
                                types_based_advice_list.append(
                                    "Turn " + self.turn_advice_dict[turn_direction] + ' immediately.')

                if types_list[i] == 'entrance_out' and 'passing_door' in types_list[i + 1]:
                    if i + 2 < len(types_list):
                        next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i + 1]]['location'],
                                                                        self.dp_dict[travel_path[i + 2]]['location'])
                    if next_direction != curr_heading:
                        turn_direction = self.get_turn_direction(path_heading, next_direction)
                        if turn_direction is not None:
                            types_based_advice_list.append(
                                "First, turn " + self.turn_advice_dict[turn_direction] + '.')

            if types_list[i] == 'door':
                if types_list[i + 1] == 'entrance_in':
                    # if i > 2 and [travel_path[i - 3], travel_path[i - 2], travel_path[i - 1], travel_path[i],
                    #               travel_path[i + 1]] == [100, 94, 95, 96, 97]:
                    #     types_based_advice_list.append("Enter room directly across the hallway.")
                    # elif i > 2 and [travel_path[i - 3], travel_path[i - 2], travel_path[i - 1], travel_path[i],
                    #                 travel_path[i + 1]] == [115, 109, 110, 111, 112]:
                    #     types_based_advice_list.append("Enter room directly at the end of the hallway.")
                    # else:
                    if len(types_based_advice_list) > 0 and types_based_advice_list[-1] == 'Enter room.':
                        types_based_advice_list.append("Enter room within current room.")
                    elif len(types_based_advice_list) > 0 and types_based_advice_list[-1] == 'Exit room.':
                        types_based_advice_list.append("Enter next room ahead.")
                    else:
                        types_based_advice_list.append("Enter room.")

            if types_list[i] == 'passing_door' and i > 0:
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
                    if turn_direction == BEHIND and self.dp_dict[travel_path[i+1]] in [160,161,173,174]:
                        types_based_advice_list.append(
                            "Proceed to " + self.nth_advice_dict[door_side_count[turn_direction]]
                            + " door within room.")
                    else:
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

            if types_list[i] == 'door':
                if types_list[i + 1] == 'entrance_out':
                    # if travel_path[i] in [29, 77, 80]:
                    #     types_based_advice_list.append("Exit room out of North-side door.")
                    # elif travel_path[i] in [169, 119, 118]:
                    #     types_based_advice_list.append("Exit room out of South-side door.")
                    # else:
                    #
                    if len(types_based_advice_list) > 0 and 'Exit' in types_based_advice_list[-1]:
                        types_based_advice_list.append("Exit next room as well.")
                    else:
                        types_based_advice_list.append("Exit current room.")

            if 'intersection' in types_list[i]:
                next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                                self.dp_dict[travel_path[i + 1]]['location'])
                # print("directions to go ", path_heading, next_direction)
                if travel_path[i] == 227 and travel_path[i+1] == 220:
                    print('next_direction is', next_direction)
                    next_direction = SOUTH

                turn_direction = self.get_turn_direction(path_heading, next_direction)
                # print(f'Path heading {path_heading} to next: {next_direction}, direction = {turn_direction}')
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
                    # if len(types_based_advice_list) > 0 and 'immediately' in types_based_advice_list[-1]:
                    #     skip = True
                    if len(types_based_advice_list) > 0 and 'intersection' in types_based_advice_list[-1]:
                        types_based_advice_list.append(
                            "At next intersection, turn " + self.turn_advice_dict[turn_direction] + ".")
                    else:
                        types_based_advice_list.append(
                            "At intersection, turn " + self.turn_advice_dict[turn_direction] + ".")

            if 'passing_door' in types_list[i] and i > 0:
                if types_list[i - 1] == 'entrance_out':
                    next_direction = self.get_direction_of_movement(self.dp_dict[travel_path[i]]['location'],
                                                                    self.dp_dict[travel_path[i + 1]]['location'])
                    if i == 1:
                        turn_direction = self.get_turn_direction(curr_heading, next_direction)
                    else:
                        turn_direction = self.get_turn_direction(path_heading, next_direction)

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
        generated_advice = " \n".join(types_based_advice_list[:min(3, len(types_based_advice_list))])

        # self.level_2_static_hold_counter += 1
        # if self.level_2_static_hold_counter >= 5:
        #     self.level_2_static_hold_counter = 0
        #     self.level_2_hold_advice = generated_advice
        #     output = generated_advice
        # else:
        #     output = self.level_2_hold_advice
        output = generated_advice
        final_destination = None
        return output, final_destination

    def update_victims_record(self, victim_save_record):
        if self.num_saved != len(victim_save_record):
            for index in victim_save_record:
                loc = (victim_save_record[index]['x'], victim_save_record[index]['y'])
                self.victim_path_details[self.location_to_victim_path_idx[loc]]['saved_state'] = True
            self.num_saved = len(victim_save_record)

    def check_player_floodfill_bayesian_efficient_old(self, past_trajectory, player_x, player_y, victim_idx,
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
        victim_likelihood_product = 0
        # print("self.gameboard_floodfill_dictionary.keys()", self.gameboard_floodfill_dictionary.keys())

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

            prob_traj_given_goals = 1
            gameboard_floodfill = self.gameboard_floodfill_dictionary[candidate_idx]
            # print("gameboard_floodfill.shape", gameboard_floodfill.shape)

            cur_loc = past_trajectory[-1]
            prev_loc = past_trajectory[-2]
            if gameboard_floodfill[cur_loc[0], cur_loc[1]] <= gameboard_floodfill[prev_loc[0], prev_loc[1]]:
                proba = 1 - epsilon

            else:
                proba = (epsilon)
            # print("proba = ", proba)
            count_denom = 0
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
            # print("likelihood_product = ", likelihood_product)
            if candidate_idx == victim_idx:
                # previous_likelihood_dict[candidate_idx] = likelihood_product * prob_goal
                victim_likelihood_product = likelihood_product * prob_goal
                denominator += likelihood_product * prob_goal

            else:
                # previous_likelihood_dict[candidate_idx] = likelihood_product * prob_non_goal
                denominator += likelihood_product * prob_non_goal

        # check for problem
        if victim_likelihood_product == 0:
            print('problem at vic index: ', victim_idx)

        # Now compute posterior : P(traj|goal) * P(goal) / P(traj)
        # denom = sum(previous_likelihood_dict.values())
        # print('denom', denom)
        likelihood = victim_likelihood_product / denominator
        most_likely_vic_idx = 0
        most_likely_vic_posterior = 0
        for candidate_idx in self.gameboard_floodfill_dictionary.keys():
            check_cand = np.prod(self.previous_likelihood_dict[candidate_idx]) / denominator
            if check_cand > most_likely_vic_posterior:
                most_likely_vic_posterior = check_cand
                most_likely_vic_idx = candidate_idx
        #
        # print()
        # print('most_likely_vic_idx: ', most_likely_vic_idx)
        # print('actual vic: ', victim_idx)

        return likelihood

    def check_player_floodfill_bayesian_efficient(self, past_trajectory, player_x, player_y, victim_idx,
                                                  previously_saved):
        # print("id_to_goal_tuple", self.id_to_goal_tuple)
        # print('victim_idx', victim_idx)
        # print('self.previous_likelihood_dict', self.previous_likelihood_dict)
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
        # print("likelihood = ", likelihood)
        if likelihood < 0.3:
            return 1
        if likelihood > 0.8:
        #     if self.get_room(end_location[0], end_location[1]) in self.visited_rooms_list and self.get_room(end_location[0], end_location[1]) !='undefined':
        #         return 3
            if self.get_room(end_location[0], end_location[1]) == self.get_room(start_location[0], start_location[1]):
                return 3
        return 2

    def generate_adaptive_instructions(self, start_location, end_location, curr_heading, victim_save_record,
                                                 past_traj_input, next_victim_to_save_id, game_time,
                                                 to_plot=False):
        # self.run_floodfill_on_victims()
        # self.advice_counter += 1
        # print('self.advice_counter', self.advice_counter)
        past_traj = self.process_past_traj_full(past_traj_input['x'])
        self.level_2_hold_counter += 1
        previously_saved = []
        # print("self.previous_likelihood_dict", self.previous_likelihood_dict)
        for save_order in victim_save_record:
            x = victim_save_record[save_order]['x']
            y = victim_save_record[save_order]['y']
            vic_idx = self.goal_tuple_to_id[str((x, y))]
            previously_saved.append(vic_idx)

        # At 2 min, clear history
        if game_time <= 120 and game_time > 117:
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

            if self.level_2_hold_counter >= 3:
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
                self.level = self.compute_new_level(start_location, end_location, next_victim_to_save_id, past_traj,
                                                    previously_saved)
            except:
                self.level = self.level

        self.level_change_counter += 1
        self.previously_saved = previously_saved

        return output








































