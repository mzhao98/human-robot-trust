import pandas as pd
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import copy
from scipy import spatial
import queue
import sys
# import config
import heapq
import pickle as pkl
import pygame
import time
import pygame, sys, time, random
from pygame.locals import *
from a_star import Search, Decision_Point_Search
import matplotlib.cm as cm
from advice_utils import *
from naive_advice_generation import *
from bayesian_advice_generation import *
from player_class import Player
from gameboard_utils import *
from redraw_windows import *




def main_Floodfill(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
         reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
         goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id, id_to_goal_tuple, victim_path,
         gameboard_floodfill_dictionary, decision_points, decision_neighbors):
    run = True
    p = Player(50,50,10,10,'#DC143C', gameboard, obstacles, yellow_locs, green_locs)
    clock = pygame.time.Clock()
    current_index = 0
    remaining_allowance = 50000
    past_path = []
    past_planned_path = []

    victim_idx = victim_path[len(p.saved)]
    gameboard_floodfill = gameboard_floodfill_dictionary[victim_idx]
    # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
    prev_fill_values = []
    past_trajectory = []
    curr_advice = 'Start.'
    level = 2
    hold_level = 0
    hold_advice = 0
    previous_room = None

    previously_visited_rooms = []

    previous_likelihood_dict = {}
    for vic_id in id_to_goal_tuple:
        previous_likelihood_dict[vic_id] = [1]*40


    while run:
        # print('level', level)
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        victim_saved = p.move(win)
        if victim_saved:
            hold_level, hold_advice = 100,100

            # level = 2
            current_index = 0
            plot_gameboard = update_plot_gameboard(plot_gameboard, p, obstacles, stairs, doors, yellow_locs, green_locs)
            victim_idx = victim_path[len(p.saved)]
            gameboard_floodfill = gameboard_floodfill_dictionary[victim_idx]
            # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
            # prev_fill_values = [prev_fill_values[-1]]
            # past_trajectory = [past_trajectory[-1]]
            previous_likelihood_dict = {}
            for vic_id in id_to_goal_tuple:
                previous_likelihood_dict[vic_id] = [1] * 40
            if reversed_complete_room_dict[id_to_goal_tuple[victim_idx]] != previous_room:
                level = 2
        # p.display_text(win)
        # current_index, remaining_allowance, past_path, past_planned_path = redrawWindow(win, p, gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, position_to_room_dict, current_index, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, remaining_allowance, past_path, past_planned_path)
        # current_index, prev_fill_values, past_trajectory = redrawWindow_bayesian_simple(win, p, gameboard,
        #                         obstacles, yellow_locs, green_locs, plot_gameboard,
        #                          reversed_complete_room_dict, position_to_room_dict, current_index, all_paths,
        #                          goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2,
        #                          goal_to_goal_instruction_level_1, gameboard_floodfill, prev_fill_values,
        #                          gameboard_floodfill_dictionary, past_trajectory,
        #                          victim_idx, victim_path, doors, stairs, goal_tuple_to_id, id_to_goal_tuple)
        # current_index, prev_fill_values, past_trajectory, curr_advice, level, hold_level, hold_advice = redrawWindow_bayesian_floodfill(win,
        #                                 p, gameboard, obstacles, yellow_locs, green_locs, plot_gameboard,
        #                                 reversed_complete_room_dict, position_to_room_dict, current_index, all_paths,
        #                                 goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2,
        #                                 goal_to_goal_instruction_level_1, gameboard_floodfill, prev_fill_values,
        #                                 gameboard_floodfill_dictionary, past_trajectory,
        #                                 victim_idx, victim_path, doors, stairs, goal_tuple_to_id, id_to_goal_tuple,
        #                                 curr_advice, level, hold_level, hold_advice, decision_points, decision_neighbors)
        current_index, prev_fill_values, past_trajectory, curr_advice, level, hold_level, hold_advice, previous_likelihood_dict, previously_visited_rooms, previous_room = redrawWindow_bayesian_floodfill_efficient(
                                                                        win, p, gameboard, obstacles, yellow_locs, green_locs, plot_gameboard,
                                                                        reversed_complete_room_dict, position_to_room_dict, current_index, all_paths,
                                                                        goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2,
                                                                        goal_to_goal_instruction_level_1, gameboard_floodfill, prev_fill_values,
                                                                        gameboard_floodfill_dictionary, past_trajectory,
                                                                        victim_idx, victim_path, doors, stairs, goal_tuple_to_id, id_to_goal_tuple,
                                                                        curr_advice, level, hold_level, hold_advice, decision_points, decision_neighbors,
                                                                        previous_likelihood_dict, previously_visited_rooms, previous_room)

def main_Value_Iteration(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
         reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
         goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id, id_to_goal_tuple, victim_path,
         gameboard_values_dictionary, gameboard_rewards_dictionary):
    run = True
    p = Player(50,50,10,10,'#DC143C', gameboard, obstacles, yellow_locs, green_locs)
    clock = pygame.time.Clock()
    current_index = 0
    remaining_allowance = 50000
    past_path = []
    past_planned_path = []

    victim_idx = victim_path[len(p.saved)]
    gameboard_floodfill = gameboard_values_dictionary[victim_idx]
    # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
    prev_fill_values = []
    past_trajectory = []

    while run:
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        victim_saved = p.move(win)
        if victim_saved:
            current_index = 0
            plot_gameboard = update_plot_gameboard(plot_gameboard, p, obstacles, stairs, doors, yellow_locs, green_locs)
            victim_idx = victim_path[len(p.saved)]
            gameboard_floodfill = gameboard_values_dictionary[victim_idx]
            # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
            prev_fill_values = []
            past_trajectory = []
        # p.display_text(win)
        # current_index, remaining_allowance, past_path, past_planned_path = redrawWindow(win, p, gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, position_to_room_dict, current_index, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, remaining_allowance, past_path, past_planned_path)
        current_index, prev_fill_values, past_trajectory = redrawWindow_ValueIteration(win, p, gameboard, obstacles,
                yellow_locs, green_locs, plot_gameboard,
                 reversed_complete_room_dict, position_to_room_dict, current_index, all_paths,
                 goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1,
                 gameboard_floodfill, prev_fill_values, gameboard_values_dictionary,
                 gameboard_rewards_dictionary, past_trajectory, victim_idx, victim_path,
                                doors, stairs, goal_tuple_to_id, id_to_goal_tuple)

def main_Naive(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
         reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
         goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id, id_to_goal_tuple, victim_path,
         gameboard_values_dictionary, gameboard_rewards_dictionary):
    run = True
    p = Player(50,50,10,10,'#DC143C', gameboard, obstacles, yellow_locs, green_locs)
    clock = pygame.time.Clock()
    current_index = 0
    remaining_allowance = 50000
    past_path = []
    past_planned_path = []

    victim_idx = victim_path[len(p.saved)]
    gameboard_floodfill = gameboard_values_dictionary[victim_idx]
    # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
    prev_fill_values = []
    past_trajectory = []

    while run:
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        victim_saved = p.move(win)
        if victim_saved:

            current_index = 0
            plot_gameboard = update_plot_gameboard(plot_gameboard, p, obstacles, stairs, doors, yellow_locs, green_locs)
            victim_idx = victim_path[len(p.saved)]
            gameboard_floodfill = gameboard_values_dictionary[victim_idx]
            # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
            prev_fill_values = []
            past_trajectory = []
        # p.display_text(win)
        current_index, remaining_allowance, past_path, past_planned_path = redrawWindow_Naive(win, p,
                        gameboard, obstacles, yellow_locs, green_locs, plot_gameboard,
                       reversed_complete_room_dict, position_to_room_dict, current_index, all_paths,
                       goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2,
                       goal_to_goal_instruction_level_1, remaining_allowance, past_path, past_planned_path,
                       doors, stairs, goal_tuple_to_id, id_to_goal_tuple)

def main_Naive_EuclideanDistance(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
         reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
         goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id, id_to_goal_tuple, victim_path,
         gameboard_values_dictionary, gameboard_rewards_dictionary):
    run = True
    p = Player(50,50,10,10,'#DC143C', gameboard, obstacles, yellow_locs, green_locs)
    clock = pygame.time.Clock()
    current_index = 0
    remaining_allowance = 50000
    past_path = []
    past_planned_path = []

    victim_idx = victim_path[len(p.saved)]
    gameboard_floodfill = gameboard_values_dictionary[victim_idx]
    # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
    prev_fill_values = []
    past_trajectory = []

    while run:
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        victim_saved = p.move(win)
        if victim_saved:
            current_index = 0
            plot_gameboard = update_plot_gameboard(plot_gameboard, p, obstacles, stairs, doors, yellow_locs, green_locs)
            victim_idx = victim_path[len(p.saved)]
            gameboard_floodfill = gameboard_values_dictionary[victim_idx]
            # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
            prev_fill_values = []
            past_trajectory = []
        # p.display_text(win)
        redrawWindow_EuclideanDistance_Naive(win, p, gameboard, obstacles, yellow_locs, green_locs, plot_gameboard,
                                             reversed_complete_room_dict, position_to_room_dict, current_index,
                                             all_paths,
                                             goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2,
                                             goal_to_goal_instruction_level_1, remaining_allowance, past_path,
                                             past_trajectory,
                                             doors, stairs, goal_tuple_to_id, id_to_goal_tuple)

def main_Naive_SmartEuclideanDistance(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
         reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
         goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id, id_to_goal_tuple, victim_path,
         gameboard_values_dictionary, gameboard_rewards_dictionary):
    run = True
    p = Player(50,50,10,10,'#DC143C', gameboard, obstacles, yellow_locs, green_locs)
    clock = pygame.time.Clock()
    current_index = 0
    remaining_allowance = 50000
    past_path = []
    past_planned_path = []

    victim_idx = victim_path[len(p.saved)]
    gameboard_floodfill = gameboard_values_dictionary[victim_idx]
    # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
    prev_fill_values = []
    past_trajectory = []

    while run:
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        victim_saved = p.move(win)
        if victim_saved:
            current_index = 0
            plot_gameboard = update_plot_gameboard(plot_gameboard, p, obstacles, stairs, doors, yellow_locs, green_locs)
            victim_idx = victim_path[len(p.saved)]
            gameboard_floodfill = gameboard_values_dictionary[victim_idx]
            # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
            prev_fill_values = []
            past_trajectory = []
        # p.display_text(win)
        redrawWindow_SmartEuclideanDistance_Naive(win, p, gameboard, obstacles, yellow_locs, green_locs, plot_gameboard,
                                             reversed_complete_room_dict, position_to_room_dict, current_index,
                                             all_paths,
                                             goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2,
                                             goal_to_goal_instruction_level_1, remaining_allowance, past_path,
                                             past_trajectory,
                                             doors, stairs, goal_tuple_to_id, id_to_goal_tuple, victim_path)

def main_Naive_SmartEuclideanDistance_AllWaypoints(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
         reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
         goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id, id_to_goal_tuple, victim_path,
         gameboard_values_dictionary, gameboard_rewards_dictionary):
    run = True
    p = Player(50,50,10,10,'#DC143C', gameboard, obstacles, yellow_locs, green_locs)
    clock = pygame.time.Clock()
    current_index = 0
    remaining_allowance = 50000
    past_path = []
    past_planned_path = []

    victim_idx = victim_path[len(p.saved)]
    gameboard_floodfill = gameboard_values_dictionary[victim_idx]
    # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
    prev_fill_values = []
    past_trajectory = []
    travel_path = []
    hold_advice = 0
    hold_level = 0
    level = 2
    curr_advice = 'Start.'

    while run:
        clock.tick(20)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()

        victim_saved = p.move(win)
        if victim_saved:
            current_index = 0
            plot_gameboard = update_plot_gameboard(plot_gameboard, p, obstacles, stairs, doors, yellow_locs, green_locs)
            victim_idx = victim_path[len(p.saved)]
            gameboard_floodfill = gameboard_values_dictionary[victim_idx]
            # gameboard_floodfill = run_floodfill(gameboard, victim_idx, obstacles, save=False)
            prev_fill_values = []
            past_trajectory = []
        # p.display_text(win)


        current_index, past_trajectory, travel_path, level, curr_advice, hold_advice, hold_level = redrawWindow_SmartEuclideanDistance_AllWaypoints_Naive(win, p,
                        gameboard, obstacles, yellow_locs, green_locs, plot_gameboard,
                       reversed_complete_room_dict, position_to_room_dict, current_index, all_paths,
                       goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2,
                       goal_to_goal_instruction_level_1, remaining_allowance, travel_path, past_trajectory,
                       doors, stairs, goal_tuple_to_id, id_to_goal_tuple, victim_path, level, curr_advice, hold_advice, hold_level)

def setup_game():
    with open('setup_files/aggregate_path_22vic_2.pkl', 'rb') as file:
        victim_path = pkl.load(file)

    goal_tuple_to_id, goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix = load_goal_distances(cutoff_victims=34)

    gameboard, obstacles, yellow_locs, green_locs, doors, stairs = get_inv_gameboard()

    rooms_dictionary, position_to_room_dict, complete_room_dict, reversed_complete_room_dict = get_rooms_dict(gameboard)

    # gameboard_floodfill_dictionary = get_floodfill_dictionary(gameboard, id_to_goal_tuple, obstacles)
    # print('value iteration running...........')
    # gameboard_values_dictionary, gameboard_rewards_dictionary = get_value_iteration_dictionary(gameboard,
    #                                                                                            id_to_goal_tuple, obstacles)
    # print('value iteration done...........')

    # with open('setup_files/floodfill_dict_4.pkl', 'wb') as filename:
    #     pkl.dump(gameboard_floodfill_dictionary, filename)
    # with open('setup_files/values_dict_sink_state5_14.pkl', 'wb') as filename:
    #     pkl.dump(gameboard_values_dictionary, filename)
    # with open('setup_files/rewards_dict_sink_state5_14.pkl', 'wb') as filename:
    #     pkl.dump(gameboard_rewards_dictionary, filename)

    with open('setup_files/floodfill_dict.pkl', 'rb') as filename:
        gameboard_floodfill_dictionary = pkl.load(filename)
    with open('setup_files/values_dict_sink_state5_14.pkl', 'rb') as filename:
        gameboard_values_dictionary = pkl.load(filename)
    with open('setup_files/rewards_dict_sink_state5_14.pkl', 'rb') as filename:
        gameboard_rewards_dictionary = pkl.load(filename)

    # with open('setup_files/rewards_dict.pkl', 'rb') as filename:
    #     gameboard_rewards_dictionary = pkl.load(filename)

    with open('setup_files/all_paths_22vic_2.pkl', 'rb') as file:
        all_paths = pkl.load(file)

    yellow_drop = []
    for i in range(len(yellow_locs)):
        if (yellow_locs[i][1],yellow_locs[i][0]) not in id_to_goal_tuple.values():
            yellow_drop.append(i)
    yellow_locs = [i for j, i in enumerate(yellow_locs) if j not in yellow_drop]


    green_drop = []
    for i in range(len(green_locs)):
        if (green_locs[i][1],green_locs[i][0]) not in id_to_goal_tuple.values():
            green_drop.append(i)
    green_locs = [i for j, i in enumerate(green_locs) if j not in green_drop]

    g_width = gameboard.shape[0] * 10
    g_height = gameboard.shape[1] * 10
    plot_gameboard = np.zeros((g_width, g_height, 3))
    plot_gameboard.fill(255)
    for (x, z) in obstacles:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [25,25,112]
    for (x, z) in stairs:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [139,69,19]
    for (x, z) in doors:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [135,206,250]

    plot_gameboard[30 * 10:34 * 10, 80:90] = [135, 206, 250]

    for (z, x) in yellow_locs:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [255,215,0]
    for (z, x) in green_locs:
        for i in range(10):
            for j in range(10):
                plot_gameboard[10 * x + i, 10 * z + j] = [0, 128, 0]
    pygame.init()



    # clientNumber = 0
    #
    # black = (0, 0, 0)
    # white = (255, 255, 255)
    # red = (255, 0, 0)
    #
    # COLOR_INACTIVE = pygame.Color('lightskyblue3')
    # COLOR_ACTIVE = pygame.Color('red')
    # FONT = pygame.font.Font('freesansbold.ttf', 20)
    #
    # ROOM_FONT = pygame.font.Font('freesansbold.ttf', 13)

    goal_to_goal_instruction_level_1 = instructions_for_level_1()
    goal_to_goal_instruction_level_2 = instructions_for_level_2()
    goal_to_goal_instruction_level_3 = instructions_for_level_3(gameboard, id_to_goal_tuple, yellow_locs)

    decision_points, intersections, entrances, doors_ = decision_point_dict()
    decision_neighbors = create_decision_points_neighbors()

    return gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, \
           position_to_room_dict, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, \
           goal_to_goal_instruction_level_1, goal_tuple_to_id, victim_path, gameboard_values_dictionary, \
           gameboard_rewards_dictionary, gameboard_floodfill_dictionary, doors, stairs, \
           goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix, decision_points, decision_neighbors


def play_with_coach(coach_type='ff'):
    gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, \
    position_to_room_dict, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, \
    goal_to_goal_instruction_level_1, goal_tuple_to_id, victim_path, gameboard_values_dictionary, \
    gameboard_rewards_dictionary, gameboard_floodfill_dictionary, doors, stairs, \
    goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix, decision_points, decision_neighbors = setup_game()

    width = gameboard.shape[0] * 10
    height = gameboard.shape[1] * 10
    win = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Minecraft")

    if coach_type == 'vi':
        main_Value_Iteration(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
                             reversed_complete_room_dict, position_to_room_dict, all_paths,
                             goal_to_goal_instruction_level_3,
                             goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id,
                             id_to_goal_tuple, victim_path,
                             gameboard_values_dictionary, gameboard_rewards_dictionary)

    elif coach_type == 'ff':
        main_Floodfill(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
                       reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
                       goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id,
                       id_to_goal_tuple, victim_path,
                       gameboard_floodfill_dictionary, decision_points, decision_neighbors)

    elif coach_type == 'naive':
        main_Naive(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
                   reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
                   goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id,
                   id_to_goal_tuple, victim_path,
                   gameboard_values_dictionary, gameboard_rewards_dictionary)

    elif coach_type == 'naive_euclidean':
        main_Naive_EuclideanDistance(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
                                     reversed_complete_room_dict, position_to_room_dict, all_paths,
                                     goal_to_goal_instruction_level_3,
                                     goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1,
                                     goal_tuple_to_id, id_to_goal_tuple, victim_path,
                                     gameboard_values_dictionary, gameboard_rewards_dictionary)

    elif coach_type == 'naive_smarteuclidean':
        main_Naive_SmartEuclideanDistance(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
                                     reversed_complete_room_dict, position_to_room_dict, all_paths,
                                     goal_to_goal_instruction_level_3,
                                     goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1,
                                     goal_tuple_to_id, id_to_goal_tuple, victim_path,
                                     gameboard_values_dictionary, gameboard_rewards_dictionary)

    elif coach_type == 'naive_smarteuclidean_allwaypoints':
        main_Naive_SmartEuclideanDistance_AllWaypoints(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
                                     reversed_complete_room_dict, position_to_room_dict, all_paths,
                                     goal_to_goal_instruction_level_3,
                                     goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1,
                                     goal_tuple_to_id, id_to_goal_tuple, victim_path,
                                     gameboard_values_dictionary, gameboard_rewards_dictionary)

    else:
        main_Floodfill(win, gameboard, obstacles, doors, stairs, yellow_locs, green_locs, plot_gameboard,
                       reversed_complete_room_dict, position_to_room_dict, all_paths, goal_to_goal_instruction_level_3,
                       goal_to_goal_instruction_level_2, goal_to_goal_instruction_level_1, goal_tuple_to_id,
                       id_to_goal_tuple, victim_path,
                       gameboard_floodfill_dictionary, decision_points, decision_neighbors)


def visualize_decision_points():
    gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, \
    position_to_room_dict, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, \
    goal_to_goal_instruction_level_1, goal_tuple_to_id, victim_path, gameboard_values_dictionary, \
    gameboard_rewards_dictionary, gameboard_floodfill_dictionary, doors, stairs, \
    goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix, decision_points, decision_neighbors = setup_game()

    decision_points, intersections, entrances, doors = decision_point_dict()


    simple_map = 'setup_files/map.csv'
    simple_map_df = pd.read_csv(simple_map)

    gameboard = np.empty((50, 93, 3))
    gameboard.fill(254)

    gameboard = np.empty((50, 93, 3))
    gameboard.fill(254)

    for index, row in simple_map_df.iterrows():
        x_coor = row['x']
        z_coor = row['z']

        if row['key'] == 'walls':
            gameboard[z_coor, x_coor] = [0, 0, 0]
        elif row['key'] == 'doors':
            gameboard[z_coor, x_coor] = [102, 178, 255]
        elif row['key'] == 'stairs':
            gameboard[z_coor, x_coor] = [153, 76, 0]

    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)

    plt.imshow(gameboard.astype(np.uint64))
    # goal = goal2

    # for goal in yellow_locs:
    #     plt.scatter(goal[1], goal[0], c='y', marker='s')
    # for goal in green_locs:
    #     plt.scatter(goal[1], goal[0], c='g', marker='s')

    # for idx in range(len(aggregate_path)):
    #     plt.scatter(id_to_goal_tuple[aggregate_path[idx]][0], id_to_goal_tuple[aggregate_path[idx]][1], c='r', s=26)
    for idx in decision_points:
        # if 'door' not in decision_points[idx]['type']:
        #     continue
        loc_tuple = (decision_points[idx]['location'][0], decision_points[idx]['location'][1])
        circle = plt.Circle(loc_tuple, color='r', radius=1)

        ax.add_patch(circle)

        label = ax.annotate(str(idx), xy=(loc_tuple[0] , loc_tuple[1] ), fontsize=6)

    # print('id_to_goal_tuple', id_to_goal_tuple)
    # for idx in id_to_goal_tuple:
    #     # if 'door' not in decision_points[idx]['type']:
    #     #     continue
    #     loc_tuple = (id_to_goal_tuple[idx][0], id_to_goal_tuple[idx][1])
    #     if (loc_tuple[1], loc_tuple[0]) in yellow_locs:
    #         circle = plt.Circle(loc_tuple, color='y', radius=1)
    #     if (loc_tuple[1], loc_tuple[0]) in green_locs:
    #         circle = plt.Circle(loc_tuple, color='g', radius=1)
    #
    #     ax.add_patch(circle)
    #
    #     label = ax.annotate(str(idx), xy=(loc_tuple[0] -1, loc_tuple[1]+1), fontsize=6)

    # plt.title(savefile.split('.')[0])
    plt.savefig('numbered.png')
    # else:
    #     plt.show()
    plt.close()


def plot_empty_map():
    gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, \
    position_to_room_dict, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, \
    goal_to_goal_instruction_level_1, goal_tuple_to_id, victim_path, gameboard_values_dictionary, \
    gameboard_rewards_dictionary, gameboard_floodfill_dictionary, doors, stairs, \
    goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix, decision_points, decision_neighbors = setup_game()

    decision_points, intersections, entrances, doors = decision_point_dict()


    simple_map = 'setup_files/map.csv'
    simple_map_df = pd.read_csv(simple_map)

    gameboard = np.empty((50, 93, 3))
    #rgb(25, 94, 131)
    gameboard[:,:,0].fill(25)
    gameboard[:, :, 1].fill(94)
    gameboard[:, :, 2].fill(131)

    for index, row in simple_map_df.iterrows():
        x_coor = row['x']
        z_coor = row['z']

        if row['key'] == 'walls':
            # rgb(237, 184, 121)
            gameboard[z_coor, x_coor] = [237, 184, 121]
        elif row['key'] == 'doors':
            gameboard[z_coor, x_coor] = [185, 116, 85]
        elif row['key'] == 'stairs':
            gameboard[z_coor, x_coor] = [153, 76, 0]

    plt.figure()
    fig, ax = plt.subplots()
    ax = fig.add_subplot(111)

    plt.imshow(gameboard.astype(np.uint64))
    plt.savefig('emptymap.png')
    plt.close()

def compute_decision_point_differences():
    gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, \
    position_to_room_dict, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, \
    goal_to_goal_instruction_level_1, goal_tuple_to_id, victim_path, gameboard_values_dictionary, \
    gameboard_rewards_dictionary, gameboard_floodfill_dictionary, doors, stairs, \
    goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix, decision_points, decision_neighbors = setup_game()

    decision_points, intersections, entrances, doors = decision_point_dict()
    decision_neighbors = create_decision_points_neighbors()
    decision_distances_dict = {}
    for decision_pt_idx in decision_neighbors:
        print('decision_pt_idx', decision_pt_idx)
        start_location = decision_points[decision_pt_idx]['location']
        if start_location in obstacles:
            print('start_location', start_location)
        for neighbor in decision_neighbors[decision_pt_idx]:
            end_location = decision_points[neighbor]['location']
            if end_location in obstacles:
                print('neighbor', neighbor)
                print('end_location', end_location)
            single_search = Search(gameboard, start_location, end_location, obstacles, yellow_locs, green_locs)
            distance = len(single_search.a_star_new(start_location, end_location))
            decision_distances_dict[(decision_pt_idx, neighbor)] = distance


    with open('setup_files/decision_distances_dict.pkl', 'wb') as handle:
        pkl.dump(decision_distances_dict, handle)
    return decision_distances_dict


def generate_map_textfile():
    gameboard, obstacles, yellow_locs, green_locs, plot_gameboard, reversed_complete_room_dict, \
    position_to_room_dict, all_paths, goal_to_goal_instruction_level_3, goal_to_goal_instruction_level_2, \
    goal_to_goal_instruction_level_1, goal_tuple_to_id, victim_path, gameboard_values_dictionary, \
    gameboard_rewards_dictionary, gameboard_floodfill_dictionary, doors, stairs, \
    goal_tuple_to_id_tuplekeys, id_to_goal_tuple, distance_dict, distance_matrix, decision_points, decision_neighbors = setup_game()

    gameboard_display = np.zeros((gameboard.shape[0], gameboard.shape[1]))
    print('gameboard_display', gameboard_display.shape)
    for (x, z) in obstacles: #1
        gameboard_display[x,z] = 1
    for (x, z) in stairs: #2
        gameboard_display[x,z] = 2
    for (x, z) in doors: #3
        gameboard_display[x,z] = 3

    with open('minecraft_map_NEW_1.txt', 'a') as map_file:
        for i in range(gameboard.shape[0]):
            for j in range(gameboard.shape[1]):
                if gameboard_display[i,j] == 0:
                    map_file.write('0,')
                if gameboard_display[i,j] == 1:
                    map_file.write('1,')
                if gameboard_display[i,j] == 2:
                    map_file.write('2,')
                if gameboard_display[i,j] == 3:
                    map_file.write('3,')

            map_file.write("\n")
        map_file.write("\n")

    with open('minecraft_map_NEW_2.txt', 'a') as map_file:
        for i in range(gameboard.shape[1]):
            for j in range(gameboard.shape[0]):
                if gameboard_display[j, i] == 0:
                    map_file.write('0,')
                if gameboard_display[j, i] == 1:
                    map_file.write('1,')
                if gameboard_display[j, i] == 2:
                    map_file.write('2,')
                if gameboard_display[j, i] == 3:
                    map_file.write('3,')

            map_file.write("\n")
        map_file.write("\n")

    return


if __name__ == '__main__':
    generate_map_textfile()
    # compute_decision_point_differences()
    # path = generate_level_2_astar_decision_points(41, 6, decision_points, decision_neighbors)

    # compute_decision_point_differences()
    # visualize_decision_points()
    # plot_empty_map()
    # play_with_coach(coach_type='ff')







