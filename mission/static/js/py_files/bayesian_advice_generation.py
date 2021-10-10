import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import heapq
import pickle as pkl
import pygame, sys, time, random
from a_star import Search
from gameboard_utils import *
from advice_utils import *


def run_floodfill(gameboard, victim_idx, obstacles, id_to_goal_tuple, save=False, savefile='gameboard_floodfill.png'):
    # victim_idx = 6
    gameboard_floodfill = np.empty((gameboard.shape[0], gameboard.shape[1]))
    gameboard_floodfill.fill(100)

    # stack = [id_to_goal_tuple[victim_idx]]
    # visited = []
    gameboard_floodfill[id_to_goal_tuple[victim_idx][0], id_to_goal_tuple[victim_idx][1]] = 0

    OPEN_priQ = []
    heapq.heapify(OPEN_priQ)
    heapq.heappush(OPEN_priQ, (0, id_to_goal_tuple[victim_idx]))
    CLOSED = {}

    while not len(OPEN_priQ) == 0:
        dist, curr_location = heapq.heappop(OPEN_priQ)

        if curr_location in CLOSED:
            continue

        if (curr_location[0] + 1) < gameboard.shape[0]:
            # if (curr_location[0]+1, curr_location[1]) in CLOSED or (curr_location[0]+1, curr_location[1]) in obstacles:
            if (curr_location[0] + 1, curr_location[1]) not in CLOSED and (
            curr_location[0] + 1, curr_location[1]) not in obstacles:
                heapq.heappush(OPEN_priQ, (dist + 1, (curr_location[0] + 1, curr_location[1])))
                gameboard_floodfill[curr_location[0] + 1, curr_location[1]] = dist + 1

        if (curr_location[0] - 1) >= 0:
            # if (curr_location[0] - 1, curr_location[1]) in CLOSED or (
            # curr_location[0] - 1, curr_location[1]) in obstacles:
            if (curr_location[0] - 1, curr_location[1]) not in CLOSED and (
            curr_location[0] - 1, curr_location[1]) not in obstacles:
                heapq.heappush(OPEN_priQ, (dist + 1, (curr_location[0] - 1, curr_location[1])))
                gameboard_floodfill[curr_location[0] - 1, curr_location[1]] = dist + 1

        if (curr_location[1] + 1) < gameboard.shape[1]:
            if (curr_location[0], curr_location[1] + 1) not in CLOSED and (
            curr_location[0], curr_location[1] + 1) not in obstacles:
                # if (curr_location[0], curr_location[1]+1) in CLOSED or (
                #         curr_location[0], curr_location[1]+1) in obstacles:
                heapq.heappush(OPEN_priQ, (dist + 1, (curr_location[0], curr_location[1] + 1)))
                gameboard_floodfill[curr_location[0], curr_location[1] + 1] = dist + 1

        if (curr_location[1] - 1) >= 0:
            if (curr_location[0], curr_location[1] - 1) not in CLOSED and (
            curr_location[0], curr_location[1] - 1) not in obstacles:
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

def run_value_iteration(gameboard, victim_idx, obstacles, id_to_goal_tuple, save=False, savefile='gameboard_value_iteration.png'):
    ## KEEP TRACK OF ACTION-DIRECTIONS
    ## U, D, L, R
    ## 0 - U
    ## 1 - D
    ## 2 - L
    ## 3 - R
    gameboard_rewards = np.empty((gameboard.shape[0], gameboard.shape[1]))
    gameboard_rewards.fill(0.0)
    goal_location = id_to_goal_tuple[victim_idx]

    # for i in range(3):
    #     for j in range(3):
    #         if goal_location[0]+i >= gameboard_rewards.shape[0] or goal_location[1]+j >= gameboard_rewards.shape[1]:
    #             continue
    #         else:
    #             gameboard_rewards[goal_location[0]+i, goal_location[1]+j] = 10.0
    # for i in range(3):
    #     for j in range(3):
    #         if goal_location[0]-i < 0 or goal_location[1]-j < 0:
    #             continue
    #         else:
    #             gameboard_rewards[goal_location[0]-i, goal_location[1]-j] = 10.0
    gameboard_rewards[goal_location[0], goal_location[1]] = 100.0

    gameboard_state_action_values = np.zeros((gameboard.shape[0], gameboard.shape[1]))
    gameboard_state_action_Q = np.zeros((gameboard.shape[0], gameboard.shape[1], 5))

    gamma = 0.99
    changed = True
    counter = 0
    while changed == True:
        changed = False
        counter += 1
        for i in range(gameboard_rewards.shape[0]):
            for j in range(gameboard_rewards.shape[1]):
                possible_actions = list(range(4))
                if (i,j) == (goal_location[0], goal_location[1]):
                    possible_actions = [4]


                for action_idx in possible_actions:
                    if action_idx == 0 and j-1 >= 0:
                        if (i,j-1) in obstacles:
                            continue
                        gameboard_state_action_Q[i,j,action_idx] = gameboard_rewards[i,j-1] + (gamma * gameboard_state_action_values[i,j-1])
                    elif action_idx == 1 and j+1 < gameboard_rewards.shape[1]:
                        if (i, j +1) in obstacles:
                            continue
                        gameboard_state_action_Q[i,j,action_idx] = gameboard_rewards[i,j+1] + (gamma * gameboard_state_action_values[i,j+1])
                    elif action_idx == 2 and i-1 >= 0:
                        if (i-1, j) in obstacles:
                            continue
                        gameboard_state_action_Q[i,j,action_idx] = gameboard_rewards[i-1,j] + (gamma * gameboard_state_action_values[i-1,j])
                    elif action_idx == 3 and i+1 < gameboard_rewards.shape[0]:
                        if (i+1, j) in obstacles:
                            continue
                        gameboard_state_action_Q[i,j,action_idx] = gameboard_rewards[i+1,j] + (gamma * gameboard_state_action_values[i+1,j])
                    elif action_idx == 4:
                        # SINK STATE
                        gameboard_state_action_Q[i, j, action_idx] = 100

                new_V = max(gameboard_state_action_Q[i,j])
                if gameboard_state_action_values[i, j] != new_V:
                    changed = True
                gameboard_state_action_values[i, j] = new_V
    for (i,j) in obstacles:
        gameboard_state_action_values[i,j] = np.min(gameboard_state_action_values)

    # for i in range(gameboard_state_action_values.shape[0]):
    #     for j in range(gameboard_state_action_values.shape[1]):
    #         if gameboard_state_action_values[i, j] > 1000:
    #             gameboard_state_action_values[i, j] = 1000
    #         if gameboard_state_action_values[i, j] < -1000:
    #             gameboard_state_action_values[i, j] = -1000

    if save:
        gameboard_state_action_values_plot = copy.deepcopy(gameboard_state_action_values)
        # gameboard_state_action_values_plot.astype(np.float64)
        # for i in range(gameboard_state_action_values_plot.shape[0]):
        #     for j in range(gameboard_state_action_values_plot.shape[1]):
        #         if gameboard_state_action_values_plot[i,j] > 1000:
        #             gameboard_state_action_values_plot[i, j] = 1000
        #         if gameboard_state_action_values_plot[i,j] < -1000:
        #             gameboard_state_action_values_plot[i, j] = -1000
        plt.imshow(gameboard_state_action_values_plot.astype(np.float64))
        plt.savefig(savefile)
        plt.close()

    # gameboard_state_action_values = np.interp(gameboard_state_action_values, (gameboard_state_action_values.min(), gameboard_state_action_values.max()), (-1, 1))
    # gameboard_rewards = np.interp(gameboard_rewards, (gameboard_rewards.min(), gameboard_rewards.max()), (-1, 1))
    # print('gameboard_state_action_values', gameboard_state_action_values)
    return gameboard_state_action_values, gameboard_rewards

def get_floodfill_dictionary(gameboard, id_to_goal_tuple, obstacles):
    gameboard_floodfill_dictionary = {}
    for i in id_to_goal_tuple.keys():
        gameboard_floodfill_i = run_floodfill(gameboard, i, obstacles, id_to_goal_tuple, save=False)
        gameboard_floodfill_dictionary[i] = gameboard_floodfill_i
    return gameboard_floodfill_dictionary

def get_value_iteration_dictionary(gameboard, id_to_goal_tuple, obstacles):
    gameboard_values_dictionary = {}
    gameboard_rewards_dictionary = {}
    for i in id_to_goal_tuple.keys():
        print('i = ', i)
        gameboard_value_i, gameboard_reward_i = run_value_iteration(gameboard, i, obstacles, id_to_goal_tuple, save=True,
                                                                    savefile='setup_files/gameboard_value_iteration_sinkstate3_'+str(i)+'.png')
        gameboard_values_dictionary[i] = gameboard_value_i
        gameboard_rewards_dictionary[i] = gameboard_reward_i
    return gameboard_values_dictionary, gameboard_rewards_dictionary



def check_player_floodfill(gameboard_floodfill, prev_fill_values, player_x, player_y):
    # print('checking prev_fill_values', prev_fill_values)
    errors = 0
    # counter = 0
    for i in range(1, 10):
        # if prev_fill_values[-i+1] == prev_fill_values[-i]:
        #     continue
        if prev_fill_values[-i] > prev_fill_values[-i-1]:
            errors += 1
        # counter += 1
        # if counter > 20:
        #     break
        # elif prev_fill_values[-i+1] < prev_fill_values[-i]:
    if errors > 5:
        return False
    return True

def check_player_floodfill_bayesian_indep(gameboard_floodfill_dictionary, gameboard_floodfill_current,
                                          past_trajectory, prev_fill_values, player_x, player_y, victim_idx, victim_path, previously_saved):
    epsilon = 0.1
    beta = 0.9
    # Get the probability of the current goal and trajectory
    prob_goal = beta
    prob_traj_given_goal = 1
    gameboard_floodfill = gameboard_floodfill_dictionary[victim_idx]
    for i in range(len(past_trajectory)-15, len(past_trajectory)-1):
        cur_loc = past_trajectory[i]
        prev_loc = past_trajectory[i-1]
        if gameboard_floodfill[cur_loc[0], cur_loc[1]] < gameboard_floodfill[prev_loc[0], prev_loc[1]]:
            proba = (1-epsilon)
            count_denom = 0
            if gameboard_floodfill[cur_loc[0]-1, cur_loc[1]] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            if gameboard_floodfill[cur_loc[0]+1, cur_loc[1]] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            if gameboard_floodfill[cur_loc[0], cur_loc[1]+1] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            if gameboard_floodfill[cur_loc[0], cur_loc[1]-1] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            proba /= count_denom
        if gameboard_floodfill[cur_loc[0], cur_loc[1]] > gameboard_floodfill[prev_loc[0], prev_loc[1]]:
            proba = (epsilon)
            count_denom = 0
            if gameboard_floodfill[cur_loc[0]-1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            if gameboard_floodfill[cur_loc[0]+1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            if gameboard_floodfill[cur_loc[0], cur_loc[1]+1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            if gameboard_floodfill[cur_loc[0], cur_loc[1]-1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                count_denom += 1
            proba /= count_denom
        prob_traj_given_goal *= proba

    numerator = prob_goal * prob_traj_given_goal

    # Now compute denominator
    denominator = 0
    for candidate_idx in gameboard_floodfill_dictionary.keys():
        if candidate_idx in previously_saved:
            continue
        prob_goal = beta
        prob_traj_given_goal = 1
        gameboard_floodfill = gameboard_floodfill_dictionary[candidate_idx]
        for i in range(len(past_trajectory)-15, len(past_trajectory)-1):
            cur_loc = past_trajectory[i]
            prev_loc = past_trajectory[i - 1]
            if gameboard_floodfill[cur_loc[0], cur_loc[1]] < gameboard_floodfill[prev_loc[0], prev_loc[1]]:
                proba = (1 - epsilon)
                count_denom = 0
                if gameboard_floodfill[cur_loc[0] - 1, cur_loc[1]] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0] + 1, cur_loc[1]] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] + 1] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] - 1] < gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if count_denom == 0:
                    count_denom = 1
                proba /= count_denom
            if gameboard_floodfill[cur_loc[0], cur_loc[1]] > gameboard_floodfill[prev_loc[0], prev_loc[1]]:
                proba = (epsilon)
                count_denom = 0
                if gameboard_floodfill[cur_loc[0] - 1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0] + 1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] + 1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] - 1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if count_denom == 0:
                    count_denom = 1
                proba /= count_denom
            prob_traj_given_goal *= proba
        joint_proba = prob_goal * prob_traj_given_goal
        denominator += joint_proba

    final_probability = numerator/denominator
    print('final_probability', final_probability)
    return final_probability

def check_player_floodfill_bayesian_softmax_indep(gameboard_floodfill_dictionary, gameboard_floodfill_current,
                                                  past_trajectory, prev_fill_values, player_x, player_y, victim_idx,
                                                  victim_path, previously_saved, id_to_goal_tuple, reversed_complete_room_dict):
    epsilon_dist = euclidean((player_x, player_y), id_to_goal_tuple[victim_idx])
    if epsilon_dist < 5:
        epsilon = 0.1
    else:
        epsilon = 0.3
    beta = 1
    lookback = 5
    # Get the probability of the current goal and trajectory
    # print('reversed_complete_room_dict', reversed_complete_room_dict)
    numerator_dict = {}
    for candidate_idx in gameboard_floodfill_dictionary.keys():
        if candidate_idx in previously_saved:
            continue
        prob_goal = beta
        prob_traj_given_goal = 1
        gameboard_floodfill = gameboard_floodfill_dictionary[candidate_idx]
        for i in range(len(past_trajectory)-lookback, len(past_trajectory)):
            cur_loc = past_trajectory[i]
            prev_loc = past_trajectory[i-1]
            if gameboard_floodfill[cur_loc[0], cur_loc[1]] <= gameboard_floodfill[prev_loc[0], prev_loc[1]]:
                proba = (1-epsilon)
                count_denom = 0
                if gameboard_floodfill[cur_loc[0]-1, cur_loc[1]] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0]+1, cur_loc[1]] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1]+1] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1]-1] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                proba /= count_denom

            else:
                proba = (epsilon)
                count_denom = 0
                if gameboard_floodfill[cur_loc[0]-1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0]+1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1]+1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1]-1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                proba /= count_denom

            if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, cur_loc)] == reversed_complete_room_dict[id_to_goal_tuple[victim_idx]]:
                proba = 1-epsilon
            prob_traj_given_goal *= proba

        numerator = prob_goal * prob_traj_given_goal
        numerator_dict[candidate_idx] = numerator

    # Now compute denominator
    denominator = 0
    for candidate_idx in gameboard_floodfill_dictionary.keys():
        if candidate_idx in previously_saved:
            continue
        prob_goal = beta
        prob_traj_given_goal = 1
        gameboard_floodfill = gameboard_floodfill_dictionary[candidate_idx]
        for i in range(len(past_trajectory)-lookback, len(past_trajectory)):
            cur_loc = past_trajectory[i]
            prev_loc = past_trajectory[i - 1]
            if gameboard_floodfill[cur_loc[0], cur_loc[1]] <= gameboard_floodfill[prev_loc[0], prev_loc[1]]:
                proba = (1 - epsilon)
                count_denom = 0
                if gameboard_floodfill[cur_loc[0] - 1, cur_loc[1]] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0] + 1, cur_loc[1]] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] + 1] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] - 1] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if count_denom == 0:
                    count_denom = 1
                proba /= count_denom
            else:
                proba = (epsilon)
                count_denom = 0
                if gameboard_floodfill[cur_loc[0] - 1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0] + 1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] + 1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if gameboard_floodfill[cur_loc[0], cur_loc[1] - 1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
                    count_denom += 1
                if count_denom == 0:
                    count_denom = 1
                proba /= count_denom
            if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, cur_loc)] == reversed_complete_room_dict[id_to_goal_tuple[victim_idx]]:
                proba = 1
            prob_traj_given_goal *= proba
        joint_proba = prob_goal * prob_traj_given_goal
        denominator += joint_proba

    total = 0
    beta = 20
    for candidate_idx in numerator_dict:
        final_prob = beta*numerator_dict[candidate_idx]
        numerator_dict[candidate_idx] = np.exp(final_prob)
        total += np.exp(final_prob)

    max_candidate_idx = 0.0
    max_probability = 0.0
    for candidate_idx in numerator_dict:
        softmax_prob = numerator_dict[candidate_idx]/total
        numerator_dict[candidate_idx] = softmax_prob
        if softmax_prob > max_probability:
            max_candidate_idx = candidate_idx
            max_probability = softmax_prob

    # print('max_candidate_idx = ', max_candidate_idx)
    # print('victim_idx = ', victim_idx)
    if max_candidate_idx == victim_idx:
        return True

    return False

def check_player_floodfill_bayesian_efficient(gameboard_floodfill_dictionary, gameboard_floodfill_current,
                                                  past_trajectory, prev_fill_values, player_x, player_y, victim_idx,
                                                  victim_path, previously_saved, id_to_goal_tuple, reversed_complete_room_dict,
                                              previous_likelihood_dict):
    epsilon_dist = euclidean((player_x, player_y), id_to_goal_tuple[victim_idx])
    if epsilon_dist < 5:
        epsilon = 0.1
    else:
        epsilon = 0.3
    beta = 0.8
    lookback = 5

    prob_goal = beta
    prob_non_goal = (1-beta)/(len(id_to_goal_tuple)-len(previously_saved))
    denominator = 0
    # Get the probability of the current goal and trajectory
    # print('reversed_complete_room_dict', reversed_complete_room_dict)
    victim_likelihood_product = 0
    for candidate_idx in gameboard_floodfill_dictionary.keys():
        if candidate_idx in previously_saved:
            previous_likelihood_dict[candidate_idx] = [0]*20
            continue

        prob_traj_given_goal = 1
        gameboard_floodfill = gameboard_floodfill_dictionary[candidate_idx]

        cur_loc = past_trajectory[-1]
        prev_loc = past_trajectory[-2]
        if gameboard_floodfill[cur_loc[0], cur_loc[1]] <= gameboard_floodfill[prev_loc[0], prev_loc[1]]:
            proba = 1-epsilon

        else:
            proba = (epsilon)

        count_denom = 0
        if gameboard_floodfill[cur_loc[0] - 1, cur_loc[1]] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += 1-epsilon
        if gameboard_floodfill[cur_loc[0] + 1, cur_loc[1]] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += 1-epsilon
        if gameboard_floodfill[cur_loc[0], cur_loc[1] + 1] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += 1-epsilon
        if gameboard_floodfill[cur_loc[0], cur_loc[1] - 1] <= gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += 1-epsilon

        if gameboard_floodfill[cur_loc[0]-1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += epsilon
        if gameboard_floodfill[cur_loc[0]+1, cur_loc[1]] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += epsilon
        if gameboard_floodfill[cur_loc[0], cur_loc[1]+1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += epsilon
        if gameboard_floodfill[cur_loc[0], cur_loc[1]-1] > gameboard_floodfill[cur_loc[0], cur_loc[1]]:
            count_denom += epsilon
        proba /= count_denom

        if reversed_complete_room_dict[get_nearest_regular(reversed_complete_room_dict, cur_loc)] == reversed_complete_room_dict[id_to_goal_tuple[candidate_idx]]:
            proba = 1

        previous_likelihood_dict[candidate_idx].append(proba)
        if len(previous_likelihood_dict[candidate_idx]) > 20:
            previous_likelihood_dict[candidate_idx].pop(0)

        likelihood_product = np.prod(previous_likelihood_dict[candidate_idx])

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
    likelihood = victim_likelihood_product/denominator
    most_likely_vic_idx = 0
    most_likely_vic_posterior = 0
    for candidate_idx in gameboard_floodfill_dictionary.keys():
        check_cand = np.prod(previous_likelihood_dict[candidate_idx])/denominator
        if check_cand > most_likely_vic_posterior:
            most_likely_vic_posterior = check_cand
            most_likely_vic_idx = candidate_idx
    #
    # print()
    # print('most_likely_vic_idx: ', most_likely_vic_idx)
    # print('actual vic: ', victim_idx)

    return likelihood, previous_likelihood_dict


def check_player_value_iteration(gameboard_value_iteration_dictionary, gameboard_reward_dictionary,
                                 past_trajectory, player_x, player_y, victim_idx, victim_path, previously_saved,
                                 id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs):
    # numerator reward
    gameboard_reward_current = gameboard_reward_dictionary[victim_idx]
    gameboard_value_iteration = gameboard_value_iteration_dictionary[victim_idx]
    for i in range((gameboard_value_iteration.shape[0])):
        for j in range((gameboard_value_iteration.shape[1])):
            if gameboard_value_iteration[i, j] <= 0:
                gameboard_value_iteration[i, j] = -1
            # gameboard_value_iteration[i, j] = gameboard_value_iteration[i, j] + (-1*np.min(gameboard_value_iteration))

    past_trajectory_reward = 0
    denominator_value = 0

    for (i,j) in past_trajectory[-10:]:
        # print('gameboard_reward_current[i,j]', gameboard_reward_current[i,j])
        past_trajectory_reward += gameboard_reward_current[i,j]/1000
        denominator_value += gameboard_reward_current[i,j]/1000

    past_trajectory_reward_init = copy.deepcopy(past_trajectory_reward)
    denominator_value_init = copy.deepcopy(denominator_value)
    # numerator value
    remainder_path = recompute_path((player_x, player_y), id_to_goal_tuple[victim_idx], id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs)
    future_trajectory_value = 0
    for (i, j) in remainder_path[:]:
        future_trajectory_value += gameboard_value_iteration[i, j]/1000
        denominator_value += gameboard_reward_current[i, j]/1000
        # print('gameboard_value_iteration[i, j]', gameboard_value_iteration[i, j])
    # print('goal: past_trajectory_reward + future_trajectory_value', past_trajectory_reward + future_trajectory_value)
    numerator = np.exp(past_trajectory_reward + future_trajectory_value)
    # numerator = past_trajectory_reward + future_trajectory_value

    # denominator, total path value
    denominator = np.exp(denominator_value)
    # denominator = denominator_value

    # print('past_trajectory_reward', past_trajectory_reward)
    # print('numerator', numerator)
    # print('denominator', denominator)
    prior_on_goal = 0.8

    current_goal_likelihood = (numerator/denominator)

    prior_other_goals = (1.0 - prior_on_goal) / (
                len(gameboard_value_iteration_dictionary.keys()) - 1 - len(previously_saved))

    # Marginalize over other goals
    # likelihood_data = 0
    # for candidate_idx in gameboard_value_iteration_dictionary.keys():
    #     if candidate_idx in previously_saved:
    #         continue
    #     # if candidate_idx == victim_idx:
    #     #     prior_on_curr_goal = prior_on_goal
    #     # else:
    #     #     prior_on_curr_goal = (1 - prior_on_goal)/(len(gameboard_value_iteration_dictionary.keys()) - len(previously_saved)-1)
    #     gameboard_reward_current = gameboard_reward_dictionary[candidate_idx]
    #     gameboard_value_iteration = gameboard_value_iteration_dictionary[candidate_idx]
    #
    #     # print('gameboard_value_iteration', gameboard_value_iteration)
    #     past_trajectory_reward = 0
    #     denominator_value = 0
    #
    #     # for (i, j) in past_trajectory[-10:]:
    #     #     past_trajectory_reward += gameboard_reward_current[i, j]
    #     #     denominator_value += gameboard_value_iteration[i, j]
    #     #     # print('gameboard_value_iteration[i, j]', gameboard_value_iteration[i, j])
    #
    #     past_trajectory_reward, denominator_value = past_trajectory_reward_init, denominator_value_init
    #
    #     # numerator value
    #     remainder_path = recompute_path((player_x, player_y), id_to_goal_tuple[candidate_idx], id_to_goal_tuple, gameboard,
    #                                     obstacles, yellow_locs, green_locs)
    #     future_trajectory_value = 0
    #     for (i, j) in remainder_path[:]:
    #         future_trajectory_value += gameboard_value_iteration[i, j]/1000
    #         denominator_value += gameboard_value_iteration[i, j]/1000
    #     # print('future_trajectory_value', future_trajectory_value)
    #     # print('past_trajectory_reward', past_trajectory_reward)
    #     # print('future_trajectory_value', future_trajectory_value)
    #     #
    #     # print('past_trajectory_reward + future_trajectory_value', past_trajectory_reward + future_trajectory_value)
    #     # numerator = np.exp(past_trajectory_reward + future_trajectory_value)
    #     # print('past_trajectory_reward + future_trajectory_value', past_trajectory_reward + future_trajectory_value)
    #     numerator = np.exp(past_trajectory_reward + future_trajectory_value)
    #     # print('numerator', numerator)
    #     # denominator, total path value
    #     # print('denominator_value', denominator_value)
    #     # denominator = np.exp(denominator_value)
    #     denominator = np.exp(denominator_value)
    #     # print('denominator', denominator)
    #
    #     goal_likelihood = (numerator / denominator)
    #     # print('goal_likelihood', goal_likelihood)
    #     likelihood_data += (goal_likelihood * prior_other_goals)

    likelihood_data = 1
    prob_goal_given_data = (current_goal_likelihood * prior_on_goal)/likelihood_data
    print('likelihood', current_goal_likelihood)
    # print('likelihood_data', likelihood_data)
    # print('prob_goal_given_data', prob_goal_given_data)
    # if prob_goal_given_data < 0.00000001:
    #     prob_goal_given_data = 0
    # if prob_goal_given_data > 1:
    #     prob_goal_given_data = 1
    return prob_goal_given_data

def check_player_value_iteration_newfail(gameboard_value_iteration_dictionary, gameboard_reward_dictionary,
                                 past_trajectory, player_x, player_y, victim_idx, victim_path, previously_saved,
                                 id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs):
    # numerator reward
    gameboard_reward_current = gameboard_reward_dictionary[victim_idx]
    gameboard_value_iteration = gameboard_value_iteration_dictionary[victim_idx]
    # for i in range((gameboard_value_iteration.shape[0])):
    #     for j in range((gameboard_value_iteration.shape[1])):
    #         gameboard_value_iteration[i, j] = gameboard_value_iteration[i, j] + (-1*np.min(gameboard_value_iteration))

    past_trajectory_reward = 0
    denominator_value = 0
    print('victim at ', id_to_goal_tuple[victim_idx])
    for (i,j) in past_trajectory[-10:]:
        # print('gameboard_reward_current[i,j]', gameboard_reward_current[i,j])
        past_trajectory_reward += gameboard_reward_current[i,j]
        denominator_value += gameboard_reward_current[i,j]

    # numerator value
    # remainder_path = recompute_path((player_x, player_y), id_to_goal_tuple[victim_idx], id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs)
    future_trajectory_value = 0
    reached_victim = False
    (prev_i, prev_j) = past_trajectory[-1]
    print('started at', (prev_i, prev_j))
    print('gameboard_value_iteration', gameboard_value_iteration.shape)
    while reached_victim == False:
        # print('victim search')
        # print(prev_i, prev_j)
        if (prev_i, prev_j)==id_to_goal_tuple[victim_idx]:
            break
        highest_val = -10000000
        best_i, best_j = prev_i, prev_j
        if prev_i - 1 >= 0:
            if ( prev_j , prev_i-1) == id_to_goal_tuple[victim_idx]:
                reached_victim = True
                best_i, best_j = prev_i-1, prev_j
                continue
            if (prev_i-1, prev_j) not in obstacles:
                if gameboard_value_iteration[prev_j, prev_i - 1] >= highest_val:
                    highest_val = gameboard_value_iteration[prev_j, prev_i - 1]
                    best_i, best_j = prev_i-1, prev_j
        if prev_i + 1 < gameboard_value_iteration.shape[1]:
            if (prev_i+1, prev_j ) == id_to_goal_tuple[victim_idx]:
                reached_victim = True
                best_i, best_j = prev_i+1, prev_j
                continue
            if (prev_j, prev_i + 1) not in obstacles:
                if gameboard_value_iteration[prev_j, prev_i + 1] >= highest_val:
                    highest_val = gameboard_value_iteration[prev_j, prev_i + 1]
                    best_i, best_j = prev_i+1, prev_j
        if prev_j - 1 >= 0:
            if (prev_i, prev_j - 1) == id_to_goal_tuple[victim_idx]:
                reached_victim = True
                best_i, best_j = prev_i, prev_j - 1
                continue
            if (prev_j-1, prev_i) not in obstacles:
                if gameboard_value_iteration[prev_j-1, prev_i] >= highest_val:
                    highest_val = gameboard_value_iteration[prev_j-1, prev_i]
                    best_i, best_j = prev_i, prev_j-1
        if prev_j + 1 < gameboard_value_iteration.shape[0]:
            if (prev_i, prev_j + 1) == id_to_goal_tuple[victim_idx]:
                reached_victim = True
                best_i, best_j = prev_i, prev_j + 1
                continue
            if (prev_j + 1, prev_i) not in obstacles:
                if gameboard_value_iteration[prev_j+1, prev_i] >= highest_val:
                    highest_val = gameboard_value_iteration[prev_j + 1, prev_i]
                    best_i, best_j = prev_i, prev_j+1
        if (best_i, best_j)==id_to_goal_tuple[victim_idx]:
            reached_victim = True
        prev_i, prev_j = best_i, best_j
        future_trajectory_value += gameboard_value_iteration[best_i, best_j]
        denominator_value += gameboard_reward_current[best_i, best_j]
        # print('gameboard_value_iteration[i, j]', gameboard_value_iteration[i, j])

    numerator = np.exp(past_trajectory_reward + future_trajectory_value)
    # numerator = past_trajectory_reward + future_trajectory_value

    # denominator, total path value
    denominator = np.exp(denominator_value)
    # denominator = denominator_value

    # print('past_trajectory_reward', past_trajectory_reward)
    # print('numerator', numerator)
    # print('denominator', denominator)

    current_goal_likelihood = numerator/denominator
    prior_on_goal = 0.8
    prior_other_goals = (1.0-prior_on_goal)/(len(gameboard_value_iteration_dictionary.keys())-1-len(previously_saved))

    print('current_goal_likelihood', current_goal_likelihood)

    # Marginalize over other goals
    # likelihood_data = 0
    # for candidate_idx in gameboard_value_iteration_dictionary.keys():
    #     if candidate_idx in previously_saved:
    #         continue
    #     if candidate_idx == victim_idx:
    #         prior_on_curr_goal = prior_on_goal
    #     else:
    #         prior_on_curr_goal = (1 - prior_on_goal)/(len(gameboard_value_iteration_dictionary.keys()) - len(previously_saved)-1)
    #     gameboard_reward_current = gameboard_reward_dictionary[candidate_idx]
    #     gameboard_value_iteration = gameboard_value_iteration_dictionary[candidate_idx]
    #
    #     # print('gameboard_value_iteration', gameboard_value_iteration)
    #     past_trajectory_reward = 0
    #     denominator_value = 0
    #
    #     for (i, j) in past_trajectory[-10:]:
    #         past_trajectory_reward += gameboard_reward_current[i, j]
    #         denominator_value += gameboard_value_iteration[i, j]
    #         # print('gameboard_value_iteration[i, j]', gameboard_value_iteration[i, j])
    #
    #     # numerator value
    #     future_trajectory_value = 0
    #     reached_victim = False
    #     (prev_i, prev_j) = past_trajectory[-1]
    #     while reached_victim == False:
    #         print(prev_i, prev_j)
    #         highest_val = -sys.maxsize
    #         best_i, best_j = prev_i, prev_j
    #         if prev_j - 1 >= 0:
    #             if (prev_i, prev_j - 1) not in obstacles:
    #
    #                 if gameboard_value_iteration[prev_i, prev_j - 1] > highest_val:
    #                     highest_val = gameboard_value_iteration[prev_i, prev_j - 1]
    #                     best_i, best_j = prev_i, prev_j - 1
    #         elif prev_j + 1 < gameboard_value_iteration.shape[1]:
    #             if (prev_i, prev_j + 1) not in obstacles:
    #
    #                 if gameboard_value_iteration[prev_i, prev_j + 1] > highest_val:
    #                     highest_val = gameboard_value_iteration[prev_i, prev_j + 1]
    #                     best_i, best_j = prev_i, prev_j + 1
    #         elif prev_i - 1 >= 0:
    #             if (prev_i - 1, prev_j) not in obstacles:
    #                 if gameboard_value_iteration[prev_i - 1, prev_j] > highest_val:
    #                     highest_val = gameboard_value_iteration[prev_i - 1, prev_j]
    #                     best_i, best_j = prev_i - 1, prev_j
    #         elif prev_i + 1 < gameboard_value_iteration.shape[0]:
    #             if (prev_i + 1, prev_j) not in obstacles:
    #                 if gameboard_value_iteration[prev_i + 1, prev_j] > highest_val:
    #                     highest_val = gameboard_value_iteration[prev_i + 1, prev_j]
    #                     best_i, best_j = prev_i + 1, prev_j
    #         if (best_i, best_j) == id_to_goal_tuple[candidate_idx]:
    #             reached_victim = True
    #         prev_i, prev_j = best_i, best_j
    #         future_trajectory_value += gameboard_value_iteration[best_i, best_j]
    #         denominator_value += gameboard_reward_current[best_i, best_j]
    #
    #     # print('past_trajectory_reward', past_trajectory_reward)
    #     # print('future_trajectory_value', future_trajectory_value)
    #     #
    #     # print('past_trajectory_reward + future_trajectory_value', past_trajectory_reward + future_trajectory_value)
    #     # numerator = np.exp(past_trajectory_reward + future_trajectory_value)
    #     numerator = np.exp(past_trajectory_reward + future_trajectory_value)
    #     # print('numerator', numerator)
    #     # denominator, total path value
    #     # print('denominator_value', denominator_value)
    #     # denominator = np.exp(denominator_value)
    #     denominator = np.exp(denominator_value)
    #     # print('denominator', denominator)
    #
    #     goal_likelihood = (numerator / denominator)
    #     # print('goal_likelihood', goal_likelihood)
    #     likelihood_data += (goal_likelihood * prior_other_goals)

    # print('likelihood_data', likelihood_data)
    likelihood_data = 1
    prob_goal_given_data = (current_goal_likelihood * prior_on_goal)/likelihood_data
    print('likelihood', current_goal_likelihood)
    # print('likelihood_data', likelihood_data)
    # print('prob_goal_given_data', prob_goal_given_data)
    # if prob_goal_given_data < 0.00000001:
    #     prob_goal_given_data = 0
    # if prob_goal_given_data > 1:
    #     prob_goal_given_data = 1
    return prob_goal_given_data

def check_player_value_iteration_softmax(gameboard_value_iteration_dictionary, gameboard_reward_dictionary,
                                         past_trajectory, player_x, player_y, victim_idx, victim_path,
                                         previously_saved, id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs):
    numerator_dict = {}

    for candidate_idx in gameboard_reward_dictionary:
        if candidate_idx in previously_saved:
            continue
        # numerator reward
        gameboard_reward_current = gameboard_reward_dictionary[candidate_idx]
        gameboard_value_iteration = gameboard_value_iteration_dictionary[candidate_idx]
        past_trajectory_reward = 0
        denominator_value = 0

        for (i,j) in past_trajectory[-10:]:
            past_trajectory_reward += gameboard_reward_current[i,j]
            denominator_value += gameboard_value_iteration[i,j]

        # numerator value
        remainder_path = recompute_path((player_x, player_y), id_to_goal_tuple[candidate_idx], id_to_goal_tuple, gameboard, obstacles, yellow_locs, green_locs)
        future_trajectory_value = 0
        for (i, j) in remainder_path:
            future_trajectory_value += gameboard_value_iteration[i, j]
            denominator_value += gameboard_value_iteration[i, j]

        numerator = past_trajectory_reward + future_trajectory_value

        # denominator, total path value
        denominator = denominator_value

        likelihood = numerator/denominator
        prior_on_goal = 1
        numerator_dict[candidate_idx] = likelihood

    # Marginalize over other goals
    # likelihood_data = 0
    # for candidate_idx in gameboard_value_iteration_dictionary.keys():
    #     if candidate_idx in previously_saved:
    #         continue
    #     gameboard_reward_current = gameboard_reward_dictionary[candidate_idx]
    #     gameboard_value_iteration = gameboard_value_iteration_dictionary[candidate_idx]
    #     past_trajectory_reward = 0
    #     denominator_value = 0
    #
    #     for (i, j) in past_trajectory:
    #         past_trajectory_reward += gameboard_reward_current[i, j]
    #         denominator_value += gameboard_value_iteration[i, j]
    #
    #     # numerator value
    #     remainder_path = recompute_path((player_x, player_y), id_to_goal_tuple[candidate_idx], id_to_goal_tuple, gameboard,
    #                                     obstacles, yellow_locs, green_locs)
    #     future_trajectory_value = 0
    #     for (i, j) in remainder_path:
    #         future_trajectory_value += gameboard_value_iteration[i, j]
    #         denominator_value += gameboard_value_iteration[i, j]
    #
    #     numerator = past_trajectory_reward + future_trajectory_value
    #
    #     # denominator, total path value
    #     denominator = denominator_value
    #
    #     goal_likelihood = numerator / denominator
    #     likelihood_data += goal_likelihood
    #
    # prob_goal_given_data = (likelihood * prior_on_goal)/likelihood_data
    # print('likelihood', likelihood)
    # print('likelihood_data', likelihood_data)
    # print('prob_goal_given_data', prob_goal_given_data)
    total = 0
    beta = 20
    for candidate_idx in numerator_dict:
        final_prob = beta * numerator_dict[candidate_idx]
        numerator_dict[candidate_idx] = np.exp(final_prob)
        total += np.exp(final_prob)

    max_candidate_idx = 0.0
    max_probability = 0.0
    for candidate_idx in numerator_dict:
        softmax_prob = numerator_dict[candidate_idx] / total
        numerator_dict[candidate_idx] = softmax_prob
        if softmax_prob > max_probability:
            max_candidate_idx = candidate_idx
            max_probability = softmax_prob
    #
    # print('max_candidate_idx = ', max_candidate_idx)
    # print('victim_idx = ', victim_idx)
    if max_candidate_idx == victim_idx:
        return True

    return False




