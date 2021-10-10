import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import heapq
import pickle as pkl
# from gameboard_utils import *
import sys, time, random
# sys.path.append('/Users/michellezhao/Documents/minecraft_FASTAPI/mission/js/py_files')
prefix = './mission/static/js/py_files/'
LARGE_COST = 100000000


class Node:
    color: str = "W"
    parent: tuple = (0, 0)
    g: int = sys.maxsize  # Cost var
    h: int = 0  # Heuristic var
    f: int = 0  # Combined g + h

class Node_Two:
    color: str = "W"
    parent: tuple = -1
    g: int = sys.maxsize  # Cost var
    h: int = 0  # Heuristic var
    f: int = 0  # Combined g + h


class Search:
    def __init__(self, gameboard, start, goal, obstacles, yellow_locs, green_locs):
        self.gameboard = gameboard
        self.goalFound = False
        self.rows = gameboard.shape[0]
        self.cols = gameboard.shape[1]
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.yellow_locs = yellow_locs
        self.green_locs = green_locs
        self.matrix = np.array([[Node() for i in range(self.cols)] for j in range(self.rows)])

    #         print('matrix shape', self.matrix.shape)
    #         for (r,c) in obstacles:
    #             self.setCell(r,c)

    def setCell(self, x, y):
        """Sets cell to a wall in matrix"""
        self.matrix[x][y].g = LARGE_COST

    def in_bounds(self, cur):
        """Check if cell is within bounds of grid and isn't a wall"""
        i, j = cur
        if 0 <= i < self.rows and 0 <= j < self.cols:
            return True
        else:
            return False

    def get_neighbors(self, cur):
        """Returns visitable neighbors as a list"""
        c, d = cur
        adj = [(c, d + 1), (c + 1, d), (c, d - 1), (c - 1, d)]

        #         result_adj = []
        #         for index in range(len(result_adj)):
        #             cur_loc = result_adj[index]
        #             if cur_loc[0] >= 0 and cur_loc[0] < self.rows:
        #                 if cur_loc[1] >= 0 and cur_loc[1] < self.cols:
        #                     result_adj.append(cur_loc)
        result_adj = filter(self.in_bounds, adj)
        return result_adj

    def get_cost(self, cur, to):
        """Returns cost from one cell to another. In 2D, only straight movement"""
        return 1

    # def a_star(self):
    #     # New priority queue will be used as open list for A* search. Add start node to queue and declare
    #     # empty dic for closed list
    #     OPEN_priQ = []
    #     heapq.heapify(OPEN_priQ)
    #     heapq.heappush(OPEN_priQ, (0, self.start))
    #     CLOSED = {}
    #
    #     (s_i, s_j) = self.start
    #     self.matrix[s_i][s_j].g = 0
    #
    #     # While the priority queue still has elements to process
    #     while not len(OPEN_priQ) == 0:
    #
    #         # Get the value with the lowest 'f' value and add it to the closed list
    #         # 0 = f_cost, node
    #         (i, j) = heapq.heappop(OPEN_priQ)[1]
    #         #             print('goal', goal)
    #         #             print((i,j))
    #         CLOSED[(i, j)] = True
    #
    #         # Exit if goal node found
    #         if (i, j) == self.goal:
    #             self.goalFound = True
    #             break
    #
    #         # Visit 4-directions from current cell (within boundaries)
    #         for v in self.get_neighbors((i, j)):
    #             row = v[0]
    #             col = v[1]
    #
    #             # Skip cell if already processed in closed list or marked as wall
    #             if (row, col) in CLOSED or (row, col) in obstacles:
    #                 continue
    #
    #             # Calculate the new cost from current cell to next
    #             n_cost = self.matrix[i][j].g + self.get_cost((i, j), v)
    #             #                 print("row", row)
    #             #                 print('col', col)
    #             #                 print("len", len(self.matrix[row]))
    #             cur_cost = self.matrix[row][col].g
    #
    #             # Calculate h and f if not visited yet or new cost is less then the current cost
    #             if cur_cost == LARGE_COST or n_cost < cur_cost:
    #                 # Update cost of current cell and calculate h and f
    #                 self.matrix[row][col].g = n_cost
    #                 h = self.heuristic((row, col), self.goal)
    #                 f = n_cost + h
    #
    #                 # Push this cell onto the priority queue with f as its priority and set parent
    #                 heapq.heappush(OPEN_priQ, (f, (row, col)))
    #                 self.matrix[row][col].parent = (i, j)
    #
    #     # After A* algorithm has completed, backtrack to the start to highlight the path taken
    #     shortest_path = self.backtrack(self.goal)
    #     return shortest_path

    def heuristic(self, cur, goal):
        # Manhattan distance heuristic
        #         goal = (goal[1], goal[0])
        #         print(cur, goal)

        distance = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
        return distance


    def backtrack(self, cur_input):
        cur = (cur_input[0], cur_input[1])
        shortest_path = []
        shortest_path.insert(0, (cur[0], cur[1]))
        while cur != self.start:
            cur = self.matrix[cur[0]][cur[1]].parent
            shortest_path.insert(0, (cur[0], cur[1]))
        #             self.gridQ.put((cur[0], cur[1], config.blue))
        shortest_path.insert(0, (cur[0], cur[1]))
        return shortest_path

    def a_star_new(self, start, goal):
        self.start = start
        self.goal = goal
        # New priority queue will be used as open list for A* search. Add start node to queue and declare
        # empty dic for closed list
        priQ = []
        heapq.heapify(priQ)
        heapq.heappush(priQ, (0, self.start))
        closed = {}

        s_i, s_j = self.start
        self.matrix[s_i][s_j].g = 0

        # While the priority queue still has elements to process
        while not len(priQ) == 0:

            # Get the value with the lowest 'f' value and add it to the closed list
            i, j = heapq.heappop(priQ)[1]
            closed[(i, j)] = True

            # Add this node to be colored in on pygame grid
            #             self.gridQ.put((i, j, config.red))

            # Exit if goal node found
            if (i, j) == self.goal:
                self.goalFound = True
                break

            # Visit 8-directions from current cell (within boundaries)
            for v in self.get_neighbors((i, j)):
                row = v[0]
                col = v[1]

                # Skip cell if already processed in closed list or marked as wall
                if (row, col) in closed or (row, col) in self.obstacles:
                    continue

                # Calculate the new cost from current cell to next
                n_cost = self.matrix[i][j].g + self.get_cost((i, j), v)
                cur_cost = self.matrix[row][col].g

                # Calculate h and f if not visited yet or new cost is less then the current cost
                if cur_cost == sys.maxsize or n_cost < cur_cost:
                    # Update cost of current cell and calculate h and f
                    self.matrix[row][col].g = n_cost
                    h = self.heuristic((row, col), self.goal)
                    f = n_cost + h

                    # Push this cell onto the priority queue with f as its priority and set parent
                    heapq.heappush(priQ, (f, (row, col)))
                    self.matrix[row][col].parent = (i, j)

                    # Color in cell as being processed
        #                     self.gridQ.put((row, col, config.green))

        # After A* algorithm has completed, backtrack to the start to highlight the path taken
        path = self.backtrack(self.goal)
        return path


def naive_in_order_a_star(gameboard, start, obstacles, yellow_locs, green_locs):
    # New priority queue will be used as open list for A* search. Add start node to queue and declare
    # empty dic for closed list
    LARGE_COST = 100000
    start_pos = start
    aggregate_path = []
    saved = []

    all_locs = yellow_locs
    all_locs.extend(green_locs)

    for y_iter in range(len(all_locs)):

        best_yellow_path = None
        best_yellow_idx = None
        min_yellow_path_len = LARGE_COST

        for y_idx in range(len(all_locs)):

            if y_idx in saved:
                continue

            goal = (all_locs[y_idx][1], all_locs[y_idx][0])
            gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
            single_search = Search(gameboard, start_pos, goal, obstacles, yellow_locs, green_locs)
            shortest_path = single_search.a_star_new()

            if len(shortest_path) < min_yellow_path_len:
                min_yellow_path_len = len(shortest_path)
                best_yellow_path = shortest_path
                best_yellow_idx = y_idx

        saved.append(best_yellow_idx)
        aggregate_path.extend(best_yellow_path)
        start_pos = (all_locs[best_yellow_idx][1], all_locs[best_yellow_idx][0])

    return aggregate_path


def get_distance_to_remaining_yellow_goals(goal, remaining_yellow_locs, goal_distances):
    total_dist = 0
    for goal_elem_key in goal_distances:
        goal_elem_left_node = (int(goal_elem_key.split(':')[1].split(',')[0].split('(')[1]),
                               int(goal_elem_key.split(':')[1].split(',')[1].split(')')[0]))
        goal_elem_right_node = (int(goal_elem_key.split(':')[0].split(',')[0].split('(')[1]),
                                int(goal_elem_key.split(':')[0].split(',')[1].split(')')[0]))
        edge_weight = goal_distances[goal_elem_key]

        if goal_elem_left_node != goal or goal_elem_right_node == goal:
            continue
        #         print('right',goal_elem_right_node)
        #         print('remaining_yellow_locs', remaining_yellow_locs)
        if goal_elem_right_node in remaining_yellow_locs:
            total_dist += edge_weight
    return total_dist


def naive_in_order_w_time_a_star(gameboard, start, obstacles, yellow_locs, green_locs, goal_distances):
    # New priority queue will be used as open list for A* search. Add start node to queue and declare
    # empty dic for closed list
    LARGE_COST = 100000
    start_pos = start
    aggregate_path = []
    saved = []
    saved_locs = []
    remaining_yellows = []

    all_locs = yellow_locs
    all_locs.extend(green_locs)

    for y_iter in range(len(all_locs)):
        #         remaining_goals = np.setdiff1d(range(len(all_locs)), saved)
        remaining_y_goals = []
        for r in range(len(yellow_locs)):

            if (yellow_locs[r][1], yellow_locs[r][0]) not in saved_locs:
                remaining_y_goals.append((yellow_locs[r][0], yellow_locs[r][1]))
        print('len remaining_y_goals', len(remaining_y_goals))

        #         new_cand_goal = (all_locs[best_yellow_idx][1], all_locs[best_yellow_idx][0])
        #         print('len remaining_yellow_locs', len(remaining_y_goals))
        #         print('distance remaining', get_distance_to_remaining_yellow_goals(start_pos, remaining_y_goals, goal_distances))
        #         print('allowance = ', 200 - len(aggregate_path))
        if len(remaining_y_goals) == 0 or len(aggregate_path) > 400:
            print('selecting from all 1')
            best_yellow_path = None
            best_yellow_idx = None
            min_yellow_path_len = LARGE_COST
            print('selecting from all')
            for y_idx in range(len(all_locs)):
                goal = (all_locs[y_idx][1], all_locs[y_idx][0])

                if y_idx in saved or goal in green_locs:
                    continue

                goal = (all_locs[y_idx][1], all_locs[y_idx][0])
                gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
                single_search = Search(gameboard, start_pos, goal, obstacles, yellow_locs, green_locs)
                shortest_path = single_search.a_star_new()

                if len(shortest_path) < min_yellow_path_len:
                    min_yellow_path_len = len(shortest_path)
                    best_yellow_path = shortest_path
                    best_yellow_idx = y_idx


        elif 200 - len(aggregate_path) > get_distance_to_remaining_yellow_goals(start_pos, remaining_y_goals,
                                                                                goal_distances):
            best_yellow_path = None
            best_yellow_idx = None
            min_yellow_path_len = LARGE_COST
            print('selecting from all')
            for y_idx in range(len(all_locs)):

                goal = (all_locs[y_idx][1], all_locs[y_idx][0])

                if y_idx in saved:
                    continue

                gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
                single_search = Search(gameboard, start_pos, goal, obstacles, yellow_locs, green_locs)
                shortest_path = single_search.a_star_new()

                if len(shortest_path) < min_yellow_path_len:
                    min_yellow_path_len = len(shortest_path)
                    best_yellow_path = shortest_path
                    best_yellow_idx = y_idx

        else:
            print('selecting from yellow')

            best_yellow_path = None
            best_yellow_idx = None
            min_yellow_path_len = LARGE_COST

            for y_idx in range(len(all_locs)):

                goal = (all_locs[y_idx][0], all_locs[y_idx][1])

                #                 if goal in saved or goal in green_locs:
                #                     continue
                if goal not in remaining_y_goals:
                    #                     print('not in ')
                    #                     print('goal', goal)
                    #                     print('remaining_y_goals', remaining_y_goals)
                    continue

                goal = (all_locs[y_idx][1], all_locs[y_idx][0])
                gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
                single_search = Search(gameboard, start_pos, goal, obstacles, yellow_locs, green_locs)
                shortest_path = single_search.a_star_new()

                if len(shortest_path) < min_yellow_path_len:
                    min_yellow_path_len = len(shortest_path)
                    best_yellow_path = shortest_path
                    best_yellow_idx = y_idx

        saved.append(best_yellow_idx)
        saved_locs.append((all_locs[best_yellow_idx][1], all_locs[best_yellow_idx][0]))
        aggregate_path.extend(best_yellow_path)
        start_pos = (all_locs[best_yellow_idx][1], all_locs[best_yellow_idx][0])

    return aggregate_path


def naive_yellow_first_a_star(gameboard, start, obstacles, yellow_locs, green_locs):
    # New priority queue will be used as open list for A* search. Add start node to queue and declare
    # empty dic for closed list
    LARGE_COST = 100000
    start_pos = start
    aggregate_path = []
    saved_yellows = []
    saved_greens = []

    for y_iter in range(len(yellow_locs)):

        best_yellow_path = None
        best_yellow_idx = None
        min_yellow_path_len = LARGE_COST

        for y_idx in range(len(yellow_locs)):

            if y_idx in saved_yellows:
                continue

            goal = (yellow_locs[y_idx][1], yellow_locs[y_idx][0])
            gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
            single_search = Search(gameboard, start_pos, goal, obstacles, yellow_locs, green_locs)
            shortest_path = single_search.a_star_new()

            if len(shortest_path) < min_yellow_path_len:
                min_yellow_path_len = len(shortest_path)
                best_yellow_path = shortest_path
                best_yellow_idx = y_idx

        saved_yellows.append(best_yellow_idx)
        aggregate_path.extend(best_yellow_path)
        start_pos = (yellow_locs[best_yellow_idx][1], yellow_locs[best_yellow_idx][0])

    #     elif len(aggregate_path) <= 600:
    for g_iter in range(len(green_locs)):

        best_green_path = None
        best_green_idx = None
        min_green_path_len = LARGE_COST

        for g_idx in range(len(green_locs)):

            if g_idx in saved_greens:
                continue

            goal = (green_locs[g_idx][1], green_locs[g_idx][0])
            gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
            single_search = Search(gameboard, start_pos, goal, obstacles, yellow_locs, green_locs)
            shortest_path = single_search.a_star_new()

            if len(shortest_path) < min_green_path_len:
                min_green_path_len = len(shortest_path)
                best_green_path = shortest_path
                best_green_idx = g_idx

        saved_greens.append(best_green_idx)
        aggregate_path.extend(best_green_path)
        start_pos = (green_locs[best_green_idx][1], green_locs[best_green_idx][0])

    return aggregate_path

def run_astar(start, goal):
    gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
    aggregate_path = naive_in_order_w_time_a_star(gameboard, start, obstacles, yellow_locs, green_locs, goal_distances)

def visualize_path(aggregate_path, filename=None):
    gameboard, obstacles, yellow_locs, green_locs = get_inv_gameboard()
    participant_df = pd.read_csv('../human_data/study1/falcon_easy_processed/participant26_results.csv')

    x_traj = participant_df['x_pos'].to_numpy()
    z_traj = participant_df['z_pos'].to_numpy()

    filename = '../human_data/study1/falconeasy/participant26.metadata'

    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    room_items_data = data[1719]
    victim_data = data[1720]
    victim_list_data = victim_data['data']['mission_victim_list']

    simple_map = '../human_data/study1/falcon_easy_processed/map.csv'
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

    for i in range(len(victim_list_data)):
        transformed_x = victim_list_data[i]['x'] + 2112
        transformed_z = victim_list_data[i]['z'] - 143
        if victim_list_data[i]['block_type'] == 'block_victim_2':
            gameboard[transformed_z, transformed_x] = [255, 204, 0]
        else:
            gameboard[transformed_z, transformed_x] = [34, 168, 20]

    plt.imshow(gameboard.astype(np.uint64))
    # goal = goal2
    for idx in range(len(aggregate_path)):
        plt.scatter(aggregate_path[idx][0], aggregate_path[idx][1], c='r', s=3)

    for goal in yellow_locs:
        plt.scatter(goal[1], goal[0], c='y', marker='s')
    for goal in green_locs:
        plt.scatter(goal[1], goal[0], c='g', marker='s')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()


class Decision_Point_Search:
    def __init__(self, edge_list, decision_points):
        self.decision_points = decision_points
        self.goalFound = False
        self.edge_list = edge_list
        self.matrix = {}
        for key in decision_points:
            self.matrix[key] = Node_Two()
        with open(prefix+'setup_files/decision_distances_dict.pkl', 'rb') as handle:
            self.decision_distances_dict = pkl.load(handle)


    def setCell(self, loc):
        """Sets cell to a wall in matrix"""
        self.matrix[loc].g = LARGE_COST


    def get_neighbors(self, cur):
        """Returns visitable neighbors as a list"""
        return self.edge_list[cur]

    def get_cost(self, cur, to):
        """Returns cost from one cell to another."""
        if (cur, to) in self.decision_distances_dict:
            cost = self.decision_distances_dict[(cur,to)]
        else:
            cost = self.decision_distances_dict[(to, cur)]
        return cost

    def heuristic(self, cur, goal):
        # Manhattan distance heuristic
        distance = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
        return distance


    def backtrack(self, cur):
        # print('cur', cur)
        # cur_loc = self.decision_points[cur]['location']
        shortest_path = []
        shortest_path.insert(0, cur)
        while cur != self.start_idx:
            cur = self.matrix[cur].parent
            # print('cur', cur)
            # cur_loc = self.decision_points[cur]['location']
            shortest_path.insert(0, cur)
        #             self.gridQ.put((cur[0], cur[1], config.blue))
        # shortest_path.insert(0, cur)
        return shortest_path

    def a_star(self, start_idx, goal_idx):
        self.start_idx = start_idx
        self.goal_idx = goal_idx
        self.start = self.decision_points[start_idx]['location']
        self.goal = self.decision_points[goal_idx]['location']
        # New priority queue will be used as open list for A* search. Add start node to queue and declare
        # empty dic for closed list
        priQ = []
        heapq.heapify(priQ)
        heapq.heappush(priQ, (0, start_idx))
        closed = {}

        s_i, s_j = self.start
        self.matrix[self.start_idx].g = 0

        # While the priority queue still has elements to process
        while not len(priQ) == 0:

            # Get the value with the lowest 'f' value and add it to the closed list
            curr_idx = heapq.heappop(priQ)[1]
            # print('curr_idx', curr_idx)
            i, j = self.decision_points[curr_idx]['location']
            closed[(i, j)] = True

            # Add this node to be colored in on pygame grid
            #             self.gridQ.put((i, j, config.red))

            # Exit if goal node found
            if (i, j) == self.goal:
                self.goalFound = True
                break

            # Visit 8-directions from current cell (within boundaries)
            # print('neighbors', self.get_neighbors(curr_idx))
            for v in self.get_neighbors(curr_idx):
                # print('v = ', v)
                v_loc = self.decision_points[v]['location']
                row = v_loc[0]
                col = v_loc[1]

                # Skip cell if already processed in closed list or marked as wall
                if (row, col) in closed:
                    continue

                # Calculate the new cost from current cell to next
                n_cost = self.matrix[curr_idx].g + self.get_cost(curr_idx, v)
                cur_cost = self.matrix[v].g


                # Calculate h and f if not visited yet or new cost is less then the current cost
                if cur_cost == sys.maxsize or n_cost < cur_cost:
                    # Update cost of current cell and calculate h and f
                    self.matrix[v].g = n_cost
                    h = self.heuristic((row, col), self.goal)
                    f = n_cost + h

                    # Push this cell onto the priority queue with f as its priority and set parent
                    heapq.heappush(priQ, (f, v))
                    self.matrix[v].parent = curr_idx

                    # Color in cell as being processed
        #                     self.gridQ.put((row, col, config.green))

        # After A* algorithm has completed, backtrack to the start to highlight the path taken
        path = self.backtrack(goal_idx)
        # path = []
        return path


class Decision_Point_Search_Augmented:
    def __init__(self, decision_points):
        self.decision_points = decision_points
        self.goalFound = False
        self.matrix = {}
        for key in decision_points:
            self.matrix[key] = Node_Two()
        with open(prefix+'setup_files/augmented_dps_distances.pkl', 'rb') as handle:
            self.decision_distances_dict = pkl.load(handle)


    def setCell(self, loc):
        """Sets cell to a wall in matrix"""
        self.matrix[loc].g = LARGE_COST


    def get_neighbors(self, cur):
        """Returns visitable neighbors as a list"""
        return self.decision_points[cur]['neighbors']

    def get_cost(self, cur, to):
        """Returns cost from one cell to another."""
        if (cur, to) in self.decision_distances_dict:
            cost = self.decision_distances_dict[(cur,to)]
        else:
            cost = self.decision_distances_dict[(to, cur)]
        return cost

    def heuristic(self, cur, goal):
        # Manhattan distance heuristic
        distance = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
        return distance


    def backtrack(self, cur):
        # print('cur', cur)
        # cur_loc = self.decision_points[cur]['location']
        shortest_path = []
        shortest_path.insert(0, cur)
        while cur != self.start_idx:
            cur = self.matrix[cur].parent
            # print('cur', cur)
            # cur_loc = self.decision_points[cur]['location']
            shortest_path.insert(0, cur)
        #             self.gridQ.put((cur[0], cur[1], config.blue))
        # shortest_path.insert(0, cur)
        return shortest_path

    def a_star(self, start_idx, goal_idx):
        self.start_idx = start_idx
        self.goal_idx = goal_idx
        self.start = self.decision_points[start_idx]['location']
        self.goal = self.decision_points[goal_idx]['location']
        # New priority queue will be used as open list for A* search. Add start node to queue and declare
        # empty dic for closed list
        priQ = []
        heapq.heapify(priQ)
        heapq.heappush(priQ, (0, start_idx))
        closed = {}

        s_i, s_j = self.start
        self.matrix[self.start_idx].g = 0

        # While the priority queue still has elements to process
        while not len(priQ) == 0:

            # Get the value with the lowest 'f' value and add it to the closed list
            curr_idx = heapq.heappop(priQ)[1]
            # print('curr_idx', curr_idx)
            i, j = self.decision_points[curr_idx]['location']
            closed[(i, j)] = True

            # Add this node to be colored in on pygame grid
            #             self.gridQ.put((i, j, config.red))

            # Exit if goal node found
            if (i, j) == self.goal:
                self.goalFound = True
                break

            # Visit 8-directions from current cell (within boundaries)
            # print('neighbors', self.get_neighbors(curr_idx))
            for v in self.get_neighbors(curr_idx):
                # print('v = ', v)
                v_loc = self.decision_points[v]['location']
                row = v_loc[0]
                col = v_loc[1]

                # Skip cell if already processed in closed list or marked as wall
                if (row, col) in closed:
                    continue

                # Calculate the new cost from current cell to next
                n_cost = self.matrix[curr_idx].g + self.get_cost(curr_idx, v)
                cur_cost = self.matrix[v].g


                # Calculate h and f if not visited yet or new cost is less then the current cost
                if cur_cost == sys.maxsize or n_cost < cur_cost:
                    # Update cost of current cell and calculate h and f
                    self.matrix[v].g = n_cost
                    h = self.heuristic((row, col), self.goal)
                    f = n_cost + h

                    # Push this cell onto the priority queue with f as its priority and set parent
                    heapq.heappush(priQ, (f, v))
                    self.matrix[v].parent = curr_idx

                    # Color in cell as being processed
        #                     self.gridQ.put((row, col, config.green))

        # After A* algorithm has completed, backtrack to the start to highlight the path taken
        path = self.backtrack(goal_idx)
        # path = []
        return path

class Decision_Point_Search_Augmented_Map2:
    def __init__(self, decision_points):
        self.decision_points = decision_points
        self.goalFound = False
        self.matrix = {}
        for key in decision_points:
            self.matrix[key] = Node_Two()
        with open(prefix + 'setup_files/augmented_dps_distances_map2.pkl', 'rb') as handle:
            self.decision_distances_dict = pkl.load(handle)

    def setCell(self, loc):
        """Sets cell to a wall in matrix"""
        self.matrix[loc].g = LARGE_COST

    def get_neighbors(self, cur):
        """Returns visitable neighbors as a list"""
        return self.decision_points[cur]['neighbors']

    def get_cost(self, cur, to):
        """Returns cost from one cell to another."""
        if (cur, to) in self.decision_distances_dict:
            cost = self.decision_distances_dict[(cur, to)]
        else:
            cost = self.decision_distances_dict[(to, cur)]
        return cost

    def heuristic(self, cur, goal):
        # Manhattan distance heuristic
        distance = abs(cur[0] - goal[0]) + abs(cur[1] - goal[1])
        return distance

    def backtrack(self, cur):
        # print('cur', cur)
        # cur_loc = self.decision_points[cur]['location']
        shortest_path = []
        shortest_path.insert(0, cur)
        while cur != self.start_idx:
            cur = self.matrix[cur].parent
            # print('cur', cur)
            # cur_loc = self.decision_points[cur]['location']
            shortest_path.insert(0, cur)
        #             self.gridQ.put((cur[0], cur[1], config.blue))
        # shortest_path.insert(0, cur)
        return shortest_path

    def a_star(self, start_idx, goal_idx):
        self.start_idx = start_idx
        self.goal_idx = goal_idx
        self.start = self.decision_points[start_idx]['location']
        self.goal = self.decision_points[goal_idx]['location']
        # New priority queue will be used as open list for A* search. Add start node to queue and declare
        # empty dic for closed list
        priQ = []
        heapq.heapify(priQ)
        heapq.heappush(priQ, (0, start_idx))
        closed = {}

        s_i, s_j = self.start
        self.matrix[self.start_idx].g = 0

        # While the priority queue still has elements to process
        while not len(priQ) == 0:

            # Get the value with the lowest 'f' value and add it to the closed list
            curr_idx = heapq.heappop(priQ)[1]
            # print('curr_idx', curr_idx)
            i, j = self.decision_points[curr_idx]['location']
            closed[(i, j)] = True

            # Add this node to be colored in on pygame grid
            #             self.gridQ.put((i, j, config.red))

            # Exit if goal node found
            if (i, j) == self.goal:
                self.goalFound = True
                break

            # Visit 8-directions from current cell (within boundaries)
            # print('neighbors', self.get_neighbors(curr_idx))
            for v in self.get_neighbors(curr_idx):
                # print('v = ', v)
                v_loc = self.decision_points[v]['location']
                row = v_loc[0]
                col = v_loc[1]

                # Skip cell if already processed in closed list or marked as wall
                if (row, col) in closed:
                    continue

                # Calculate the new cost from current cell to next
                n_cost = self.matrix[curr_idx].g + self.get_cost(curr_idx, v)
                cur_cost = self.matrix[v].g

                # Calculate h and f if not visited yet or new cost is less then the current cost
                if cur_cost == sys.maxsize or n_cost < cur_cost:
                    # Update cost of current cell and calculate h and f
                    self.matrix[v].g = n_cost
                    h = self.heuristic((row, col), self.goal)
                    f = n_cost + h

                    # Push this cell onto the priority queue with f as its priority and set parent
                    heapq.heappush(priQ, (f, v))
                    self.matrix[v].parent = curr_idx

                    # Color in cell as being processed
        #                     self.gridQ.put((row, col, config.green))

        # After A* algorithm has completed, backtrack to the start to highlight the path taken
        path = self.backtrack(goal_idx)
        # path = []
        return path
