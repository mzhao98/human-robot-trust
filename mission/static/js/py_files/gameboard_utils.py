import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import heapq
import pickle as pkl
# import pygame, sys, time, random


import sys, time, random
# sys.path.append('./mission/static/js/py_files')

# from mission.static.js.py_files.a_star import Search, Decision_Point_Search

prefix = './mission/static/js/py_files/'


def get_gameboard():
    participant_df = pd.read_csv(prefix+'setup_files/participant26_results.csv')

    x_traj = participant_df['x_pos'].to_numpy()
    z_traj = participant_df['z_pos'].to_numpy()

    filename = prefix+'setup_files/participant26.metadata'

    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    room_items_data = data[1719]
    victim_data = data[1720]
    victim_list_data = victim_data['data']['mission_victim_list']

    simple_map = prefix+'setup_files/map.csv'
    simple_map_df = pd.read_csv(simple_map)

    gameboard = np.empty((50, 93, 3))
    gameboard.fill(254)

    gameboard = np.empty((50, 93, 3))
    gameboard.fill(254)

    obstacles = []
    doors = []
    stairs = []

    for index, row in simple_map_df.iterrows():
        x_coor = row['x']
        z_coor = row['z']

        if row['key'] == 'walls':
            gameboard[z_coor, x_coor] = [0, 0, 0]
            # obstacles.append((z_coor, x_coor))
        elif row['key'] == 'doors':
            gameboard[z_coor, x_coor] = [102, 178, 255]
        elif row['key'] == 'stairs':
            gameboard[z_coor, x_coor] = [153, 76, 0]

    for i in range(len(victim_list_data)):
        transformed_x = victim_list_data[i]['x'] + 2112
        transformed_z = victim_list_data[i]['z'] - 143
        if victim_list_data[i]['block_type'] == 'block_victim_2':
            gameboard[transformed_z, transformed_x] = [255, 204, 0]
        #             print((transformed_z, transformed_x))
        else:
            gameboard[transformed_z, transformed_x] = [34, 168, 20]
    return gameboard

def get_inv_gameboard():
    participant_df = pd.read_csv(prefix+'setup_files/participant26_results.csv')

    x_traj = participant_df['x_pos'].to_numpy()
    z_traj = participant_df['z_pos'].to_numpy()

    filename = prefix+'setup_files/participant26.metadata'

    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))

    room_items_data = data[1719]
    victim_data = data[1720]
    victim_list_data = victim_data['data']['mission_victim_list']

    simple_map = prefix+'setup_files/map.csv'
    simple_map_df = pd.read_csv(simple_map)

    gameboard = np.empty((93, 50, 3))
    gameboard.fill(254)

    gameboard = np.empty((93, 50, 3))
    gameboard.fill(254)

    obstacles = []
    doors = []
    stairs = []

    obstacles_stairs_free = []

    for index, row in simple_map_df.iterrows():
        x_coor = row['x']
        z_coor = row['z']

        if row['key'] == 'walls':
            gameboard[x_coor, z_coor] = [0,0,0]
            obstacles.append((x_coor, z_coor))
            obstacles_stairs_free.append((x_coor, z_coor))

        elif row['key'] == 'doors':
            gameboard[x_coor, z_coor] = [102,178,255]
            doors.append((x_coor, z_coor))
        elif row['key'] == 'stairs':
            gameboard[x_coor, z_coor] = [153,76,0]
            stairs.append((x_coor, z_coor))
            obstacles.append((x_coor, z_coor))
            # obstacles.append((x_coor, z_coor))

    # for i in range(gameboard.shape[0]):
    #     for j in range(gameboard.shape[1]):
    #         obstacles.append((i, j))
    # plt.figure(figsize=(15,15))
    yellow_locs = []
    green_locs = []

    for i in range(len(victim_list_data)):
        transformed_x = victim_list_data[i]['x'] + 2112
        transformed_z = victim_list_data[i]['z'] - 143
        if victim_list_data[i]['block_type'] == 'block_victim_2':
            gameboard[transformed_x, transformed_z] = [255,204,0]
            yellow_locs.append((transformed_z, transformed_x))
        else:
            gameboard[transformed_x, transformed_z] = [34,168,20]
            green_locs.append((transformed_z, transformed_x))

    # plt.imshow(gameboard.astype(np.uint64))
    # plt.show()
    return gameboard, obstacles, yellow_locs, green_locs, doors, stairs, obstacles_stairs_free

def load_goal_distances(cutoff_victims=25):
    with open(prefix+'setup_files/goal_distances2.pkl', 'rb') as file:
        goal_distances = pkl.load(file)
    goal_tuple_to_id = {}
    counter = 0

    goal_distances_keys = list(goal_distances.keys())
    #np.random.shuffle(goal_distances_keys)
    for key in goal_distances_keys:
        key1 = key.split(':')[0]
        key2 = key.split(':')[1]
        if key1 not in goal_tuple_to_id and len(goal_tuple_to_id.keys()) < cutoff_victims:
            goal_tuple_to_id[key1] = counter
            counter += 1
        if key2 not in goal_tuple_to_id and len(goal_tuple_to_id.keys()) < cutoff_victims:
            goal_tuple_to_id[key2] = counter
            counter += 1
    lst = [0, 3, 8, 9, 11, 12, 13, 15, 22, 24, 21, 29]
    for key, value in list(goal_tuple_to_id.items()):
        if value in lst:
            del goal_tuple_to_id[key]
    num_goals = len(goal_tuple_to_id.keys())

    goal_tuple_to_id_new = {}
    goal_tuple_to_id_new_string = {}
    counter = 0
    for key, value in list(goal_tuple_to_id.items()):
        goal_tuple_to_id_new[eval(key)] = counter
        goal_tuple_to_id_new_string[key] = counter
        counter += 1
    goal_tuple_to_id = copy.deepcopy(goal_tuple_to_id_new_string)
    goal_tuple_to_id_tuple = goal_tuple_to_id_new

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
    return goal_tuple_to_id, goal_tuple_to_id_tuple, id_to_goal_tuple, distance_dict, distance_matrix


def get_rooms_dict(gameboard):
    rooms_dictionary = {
        'Room 100': {'x': [24, 27], 'z': [1, 8]},
        'Open Break Area': {'x': [28, 36], 'z': [1, 8]},
        'Executive Suite 1': {'x': [37, 53], 'z': [1, 8]},
        'Executive Suite 2': {'x': [54, 68], 'z': [1, 8]},
        'King Chris\' Office 1': {'x': [69, 84], 'z': [1, 8]},
        'King Chris\' Office 2': {'x': [76, 84], 'z': [8, 16]},
        'Kings Terrace': {'x': [85, 93], 'z': [0, 17]},

        'Room 101': {'x': [76, 84], 'z': [17, 26]},
        'Room 102': {'x': [76, 84], 'z': [27, 35]},
        'Room 103': {'x': [76, 84], 'z': [41, 49]},

        'Room 104': {'x': [67, 75], 'z': [41, 49]},
        'Room 105': {'x': [58, 66], 'z': [41, 49]},
        'Room 106': {'x': [49, 57], 'z': [41, 49]},
        'Room 107': {'x': [39, 48], 'z': [41, 49]},
        'Room 108': {'x': [31, 39], 'z': [41, 49]},
        'Room 109': {'x': [22, 30], 'z': [41, 49]},
        'Room 110': {'x': [13, 21], 'z': [41, 49]},
        'Room 111': {'x': [4, 12], 'z': [41, 49]},

        'Herbalife Conference Room': {'x': [62, 70], 'z': [14, 35]},
        'Amway Conference Room': {'x': [48, 56], 'z': [14, 24]},
        'Mary Kay Conference Room': {'x': [48, 56], 'z': [25, 35]},
        'Women\'s Room': {'x': [40, 46], 'z': [27, 35]},
        'Men\'s Room': {'x': [40, 46], 'z': [18, 26]},
        'Den': {'x': [40, 46], 'z': [14, 17]},
        'The Computer Farm': {'x': [17, 34], 'z': [14, 35]},

        'Left Hallway': {'x': [17, 75], 'z': [9, 13]},
        'Right Hallway': {'x': [4, 84], 'z': [36, 40]},
        'Entrance Bridge': {'x': [35, 39], 'z': [13, 36]},
        'Middle Bridge': {'x': [57, 61], 'z': [13, 36]},
        'Rear Bridge': {'x': [71, 75], 'z': [13, 36]},

        'Women\'s Stall 1': {'x': [40, 43], 'z': [29, 31]},
        'Women\'s Stall 2': {'x': [40, 43], 'z': [33, 35]},
        'Men\'s Stall 1': {'x': [40, 43], 'z': [20, 22]},
        'Men\'s Stall 2': {'x': [40, 43], 'z': [24, 26]},
        "Starting Deck": {'x': [17, 23], 'z': [1, 7]},
    }

    position_to_room_dict = {}
    for i in range(gameboard.shape[1]):
        for j in range(gameboard.shape[0]):
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
        for i in range(rooms_dictionary[room]['z'][0], rooms_dictionary[room]['z'][1]):
            for j in range(rooms_dictionary[room]['x'][0], rooms_dictionary[room]['x'][1]):
                complete_room_dict[room].append((j, i))
    complete_room_dict['Kings Terrace'] = [(85, 0),
                                           #   (86, 0),
                                           #   (87, 0),
                                           #   (88, 0),
                                           #   (89, 0),
                                           #   (90, 0),
                                           #   (91, 0),
                                           #   (92, 0),
                                           (85, 1),
                                           #   (86, 1),
                                           #   (87, 1),
                                           #   (88, 1),
                                           #   (89, 1),
                                           #   (90, 1),
                                           #   (91, 1),
                                           #   (92, 1),
                                           (85, 2),
                                           (86, 2),
                                           (87, 2),
                                           #   (88, 2),
                                           #   (89, 2),
                                           #   (90, 2),
                                           #   (91, 2),
                                           #   (92, 2),
                                           (85, 3),
                                           (86, 3),
                                           (87, 3),
                                           (88, 3),
                                           (89, 3),
                                           #   (90, 3),
                                           #   (91, 3),
                                           #   (92, 3),
                                           (85, 4),
                                           (86, 4),
                                           (87, 4),
                                           (88, 4),
                                           (89, 4),
                                           (90, 4),
                                           #   (91, 4),
                                           #   (92, 4),
                                           (85, 5),
                                           (86, 5),
                                           (87, 5),
                                           (88, 5),
                                           (89, 5),
                                           (90, 5),
                                           (91, 5),
                                           (92, 5),
                                           (85, 6),
                                           (86, 6),
                                           (87, 6),
                                           (88, 6),
                                           (89, 6),
                                           (90, 6),
                                           (91, 6),
                                           (92, 6),
                                           (85, 7),
                                           (86, 7),
                                           (87, 7),
                                           (88, 7),
                                           (89, 7),
                                           (90, 7),
                                           (91, 7),
                                           (92, 7),
                                           (85, 8),
                                           (86, 8),
                                           (87, 8),
                                           (88, 8),
                                           (89, 8),
                                           (90, 8),
                                           (91, 8),
                                           (92, 8),
                                           (85, 9),
                                           (86, 9),
                                           (87, 9),
                                           (88, 9),
                                           (89, 9),
                                           (90, 9),
                                           (91, 9),
                                           (92, 9),
                                           (85, 10),
                                           (86, 10),
                                           (87, 10),
                                           (88, 10),
                                           (89, 10),
                                           (90, 10),
                                           (91, 10),
                                           (92, 10),
                                           (85, 11),
                                           (86, 11),
                                           (87, 11),
                                           (88, 11),
                                           (89, 11),
                                           (90, 11),
                                           (91, 11),
                                           (92, 11),
                                           (85, 12),
                                           (86, 12),
                                           (87, 12),
                                           (88, 12),
                                           (89, 12),
                                           #   (90, 12),
                                           #   (91, 12),
                                           #   (92, 12),
                                           (85, 13),
                                           (86, 13),
                                           (87, 13),
                                           (88, 13),
                                           (89, 13),
                                           #   (90, 13),
                                           #   (91, 13),
                                           #   (92, 13),
                                           (85, 14),
                                           (86, 14),
                                           (87, 14),
                                           #   (88, 14),
                                           #   (89, 14),
                                           #   (90, 14),
                                           #   (91, 14),
                                           #   (92, 14),
                                           (85, 15),
                                           (86, 15),
                                           (87, 15),
                                           (88, 15),
                                           #   (89, 15),
                                           #   (90, 15),
                                           #   (91, 15),
                                           #   (92, 15),
                                           (85, 16),
                                           #   (86, 16),
                                           #   (87, 16),
                                           #   (88, 16),
                                           #   (89, 16),
                                           #   (90, 16),
                                           #   (91, 16),
                                           #   (92, 16)
                                           ]
    reversed_complete_room_dict = {}
    for key in complete_room_dict:
        for loc in complete_room_dict[key]:
            reversed_complete_room_dict[loc] = key

            if "King Chris" in key:
                reversed_complete_room_dict[loc] = "King Chris\' Office"
    return rooms_dictionary, position_to_room_dict, complete_room_dict, reversed_complete_room_dict

def update_plot_gameboard(plot_gameboard_old, player, obstacles, stairs, doors, yellow_locs, green_locs):
    plot_gameboard = np.zeros(plot_gameboard_old.shape)
    plot_gameboard.fill(255)
    for (x, z) in obstacles:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [25, 25, 112]
    for (x, z) in stairs:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [139, 69, 19]

    for (x, z) in doors:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [135, 206, 250]

    plot_gameboard[30 * 10:34 * 10, 80:90] = [135, 206, 250]

    for (z, x) in yellow_locs:
        if (10 * x, 10 * z) in player.saved_locs:
            continue
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [255, 215, 0]
    for (z, x) in green_locs:
        if (10 * x, 10 * z) in player.saved_locs:
            continue
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [0, 128, 0]
    return plot_gameboard

