import numpy as np

from FrozenLakeEnv import FrozenLakeEnv
from typing import List, Tuple
import heapdict


def calc_results(self, first_state: int, last_state: int, parents: List[int], actions: List[int], costs: List[int],
                 expanded: int) -> Tuple[List[int], int, float]:
    total_cost = 0
    actions_output = []

    current_state = last_state
    while current_state != first_state:
        actions_output.insert(0, actions[current_state])
        total_cost += costs[current_state]
        current_state = parents[current_state]

    return actions_output, total_cost, expanded


def ucs_calc_results(self, first_state: int, last_state: int, parents: List[int], actions: List[int], costs: List[int],
                     expanded: int) -> Tuple[List[int], int, float]:
    total_cost = costs[last_state] + 1  # cost to reach final + cost of final
    actions_output = []

    current_state = last_state
    while current_state != first_state:
        actions_output.insert(0, actions[current_state])
        current_state = parents[current_state]

    return actions_output, total_cost, expanded


def minimal_open_state(open_state: List[int], costs: List[int]):
    minimal = costs[open_state[0]]
    minimal_state = open_state[0]
    for state in open_state:
        if minimal > costs[state]:
            minimal = costs[state]
            minimal_state = state
    return minimal_state


def manhattan_dist(env: FrozenLakeEnv, state1: int, state2: int):
    row1, col1 = env.to_row_col(state1)
    row2, col2 = env.to_row_col(state2)
    return abs(row1 - row2) + abs(col1 - col2)


def heuristics(env: FrozenLakeEnv, curr_state: int) -> int:
    closest_final = -1
    min_dist = 100

    for state in range(env.ncol * env.nrow):
        if env.is_final_state(state):
            if closest_final == -1 or manhattan_dist(env, state, curr_state) < min_dist:
                closest_final = state
                min_dist = manhattan_dist(env, state, curr_state)

    return min_dist


def minimal_greedy_state(env: FrozenLakeEnv, open_state: List[int], costs: List[int]):
    minimal = heuristics(env, open_state[0])
    minimal_state = open_state[0]
    for state in open_state:
        if minimal > heuristics(env, state):
            minimal = heuristics(env, state)
            minimal_state = state
    return minimal_state


def minimal_weighted_state(env: FrozenLakeEnv, open_state: List[int], costs: List[int], weight: int):
    minimal = weight * heuristics(env, open_state[0]) + (1 - weight) * costs[open_state[0]]
    minimal_state = open_state[0]
    for state in open_state:
        if minimal > (weight * heuristics(env, state) + (1 - weight) * costs[state]):
            minimal = weight * heuristics(env, state) + (1 - weight) * costs[state]
            minimal_state = state
    return minimal_state


class BFSAgent():
    def __init__(self) -> None:
        self.env = None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()

        # initialization
        visited = [0] * (self.env.ncol * self.env.nrow)
        parents = [-1] * (self.env.ncol * self.env.nrow)
        actions = [-1] * (self.env.ncol * self.env.nrow)
        costs = [-1] * (self.env.ncol * self.env.nrow)
        expanded = []

        # first node
        open_state = [state]
        parents[state] = -2

        return self.search_rec(env, visited, open_state, expanded, parents, actions, costs)

    def search_rec(self, env: FrozenLakeEnv, visited: List[int], open_state: List[int], expanded: List[int],
                   parents: List[int], actions: List[int], costs: List[int]) -> Tuple[List[int], int, float]:
        if not len(open_state):
            return [], 0, 0
        state = open_state.pop()  # new state :)
        visited[state] = 1  # i've been here

        expanded.insert(0, state)
        for action, (new_state, cost, terminated) in env.succ(state).items():  # loop over successors of new state
            if new_state is not None and visited[new_state] == 0 and parents[new_state] == -1:
                # open new state
                open_state.insert(0, new_state)

                # update parameters
                actions[new_state] = action
                costs[new_state] = cost
                parents[new_state] = state

                # check if final state
                if self.env.is_final_state(new_state):
                    return calc_results(self, env.get_initial_state(), new_state, parents, actions, costs,
                                        len(expanded))

        # recursion
        result = self.search_rec(env, visited, open_state, expanded, parents, actions, costs)
        if result != ([], 0, 0):
            return result
        return [], 0, 0


class DFSAgent():
    def __init__(self) -> None:
        self.env = None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()

        # initialization
        visited = [0] * (self.env.ncol * self.env.nrow)
        parents = [-1] * (self.env.ncol * self.env.nrow)
        actions = [-1] * (self.env.ncol * self.env.nrow)
        costs = [-1] * (self.env.ncol * self.env.nrow)
        expanded = []

        # first node
        open_state = [state]
        parents[state] = -2

        return self.search_rec(env, visited, open_state, expanded, parents, actions, costs)

    def search_rec(self, env: FrozenLakeEnv, visited: List[int], open_state: List[int], expanded: List[int],
                   parents: List[int], actions: List[int], costs: List[int]) -> Tuple[List[int], int, float]:
        if not len(open_state):
            return [], 0, 0
        state = open_state.pop()  # new state :)
        visited[state] = 1  # i've been here
        expanded.insert(0, state)
        for action, (new_state, cost, terminated) in env.succ(state).items():  # loop over successors of new state
            if new_state is not None and visited[new_state] == 0 and parents[new_state] == -1:
                # open new state
                open_state.insert(0, new_state)

                # update parameters
                actions[new_state] = action
                costs[new_state] = cost
                parents[new_state] = state
                if self.env.is_final_state(new_state):  # check if final state
                    return calc_results(self, env.get_initial_state(), new_state, parents, actions, costs,
                                        len(expanded))
                # recursion
                result = self.search_rec(env, visited, open_state, expanded, parents, actions, costs)
                if result != ([], 0, 0):
                    return result
        return [], 0, 0


class UCSAgent():

    def __init__(self) -> None:
        self.env = None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()

        # initialization
        visited = [0] * (self.env.ncol * self.env.nrow)
        parents = [-1] * (self.env.ncol * self.env.nrow)
        actions = [-1] * (self.env.ncol * self.env.nrow)
        costs = [-1] * (self.env.ncol * self.env.nrow)
        expanded = []

        # first node
        open_state = heapdict.heapdict()
        open_state[state] = (0, 0)
        parents[state] = -2

        return self.search_rec(env, visited, open_state, expanded, parents, actions, costs)

    def search_rec(self, env: FrozenLakeEnv, visited: List[int], open_state: heapdict, expanded: List[int],
                   parents: List[int], actions: List[int], costs: List[int]) -> Tuple[List[int], int, float]:
        if not len(open_state):
            return [], 0, 0
        state, k = open_state.popitem()  # new state :) and minimal!!
        visited[state] = 1  # i've been here

        # check if final state
        if self.env.is_final_state(state):
            return ucs_calc_results(self, env.get_initial_state(), state, parents, actions, costs, len(expanded))

        expanded.insert(0, state)
        for action, (new_state, cost, terminated) in env.succ(state).items():  # loop over successors of new state
            if new_state is not None and visited[new_state] == 0 and parents[new_state] == -1:
                # open new state
                open_state[new_state] = (cost + costs[state], new_state)

                # update parameters
                actions[new_state] = action
                costs[new_state] = cost + costs[state]
                parents[new_state] = state

            # if found a better path to new_state
            elif new_state is not None and parents[new_state] != -1 and costs[new_state] > (costs[state] + cost):
                parents[new_state] = state
                costs[new_state] = costs[state] + cost
                actions[new_state] = action
                open_state[new_state] = (cost + costs[state], new_state)

        # recursion
        result = self.search_rec(env, visited, open_state, expanded, parents, actions, costs)
        if result != ([], 0, 0):
            return result
        return [], 0, 0


class GreedyAgent():

    def __init__(self) -> None:
        self.env = None

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()

        # initialization
        visited = [0] * (self.env.ncol * self.env.nrow)
        parents = [-1] * (self.env.ncol * self.env.nrow)
        actions = [-1] * (self.env.ncol * self.env.nrow)
        costs = [-1] * (self.env.ncol * self.env.nrow)
        expanded = []

        # first node
        open_state = heapdict.heapdict()
        open_state[state] = (0, 0)
        parents[state] = -2

        return self.search_rec(env, visited, open_state, expanded, parents, actions, costs)

    def search_rec(self, env: FrozenLakeEnv, visited: List[int], open_state: heapdict, expanded: List[int],
                   parents: List[int], actions: List[int], costs: List[int]) -> Tuple[List[int], int, float]:
        if not len(open_state):
            return [], 0, 0
        state, k = open_state.popitem()  # new minimal greedy state + removal
        visited[state] = 1  # i've been here

        if self.env.is_final_state(state):
            return calc_results(self, env.get_initial_state(), state, parents, actions, costs,
                                len(expanded))

        expanded.insert(0, state)
        for action, (new_state, cost, terminated) in env.succ(state).items():  # loop over successors of new state
            if new_state is not None and visited[new_state] == 0 and parents[new_state] == -1:
                costs[new_state] = cost
                # open new state
                open_state[new_state] = (heuristics(env, new_state), new_state)
                # update parameters
                actions[new_state] = action
                parents[new_state] = state

                # check if final state

        # recursion
        result = self.search_rec(env, visited, open_state, expanded, parents, actions, costs)
        if result != ([], 0, 0):
            return result
        return [], 0, 0


class WeightedAStarAgent():

    def __init__(self) -> None:
        self.env = None

    def search(self, env: FrozenLakeEnv, h_weight: int) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()

        # initialization
        visited = [0] * (self.env.ncol * self.env.nrow)
        parents = [-1] * (self.env.ncol * self.env.nrow)
        actions = [-1] * (self.env.ncol * self.env.nrow)
        costs = [-1] * (self.env.ncol * self.env.nrow)
        expanded = []

        # first node
        open_state = heapdict.heapdict()
        open_state[state] = (0, 0)
        parents[state] = -2

        return self.search_rec(env, visited, open_state, expanded, parents, actions, costs, h_weight)

    def search_rec(self, env: FrozenLakeEnv, visited: List[int], open_state: heapdict, expanded: List[int],
                   parents: List[int], actions: List[int], costs: List[int], w: int) -> Tuple[List[int], int, float]:
        if not len(open_state):
            return [], 0, 0
        state, k = open_state.popitem()  # minimal weighted A* state
        visited[state] = 1  # i've been here

        if self.env.is_final_state(state):
            return ucs_calc_results(self, env.get_initial_state(), state, parents, actions, costs,
                                    len(expanded))

        expanded.insert(0, state)
        for action, (new_state, cost, terminated) in env.succ(state).items():  # loop over successors of new state
            if new_state is not None and visited[new_state] == 0 and parents[new_state] == -1:
                # open new state

                costs[new_state] = cost + costs[state]

                open_state[new_state] = (w * heuristics(env, new_state) + (1 - w) * costs[new_state], new_state)

                # update parameters
                actions[new_state] = action
                parents[new_state] = state

                # if found a better path to new_state
            elif new_state is not None and parents[new_state] != -1 and costs[new_state] > (costs[state] + cost):
                parents[new_state] = state
                costs[new_state] = costs[state] + cost
                open_state[new_state] = (w * heuristics(env, new_state) + (1 - w) * costs[new_state], new_state)
                actions[new_state] = action

        # recursion
        result = self.search_rec(env, visited, open_state, expanded, parents, actions, costs, w)
        if result != ([], 0, 0):
            return result
        return [], 0, 0


class IDAStarAgent():
    def __init__(self) -> None:
        self.env = None
        self.new_limit = np.inf

    def search(self, env: FrozenLakeEnv) -> Tuple[List[int], int, float]:
        self.env = env
        self.env.reset()
        state = self.env.get_initial_state()
        self.new_limit = heuristics(env, state) - 1

        while True:

            # initialization
            visited = [0] * (self.env.ncol * self.env.nrow)
            parents = [-1] * (self.env.ncol * self.env.nrow)
            actions = [-1] * (self.env.ncol * self.env.nrow)
            costs = [-1] * (self.env.ncol * self.env.nrow)
            expanded = []

            # first node
            parents[state] = -2

            # IDA*
            f_limit = self.new_limit
            self.new_limit = np.inf
            path = [state]
            result = self.DFS_f(env, visited, expanded, parents, actions, costs, f_limit, state, path, 0)
            if result != ([], 0, 0):
                return result

    def DFS_f(self, env: FrozenLakeEnv, visited: List[int], expanded: List[int],
              parents: List[int], actions: List[int], costs: List[int], f_limit: float, curr_state: int,
              path: List[int], g: int) -> Tuple[List[int], int, float]:

        visited[curr_state] = 1  # i've been here
        expanded.insert(0, curr_state)

        new_f = g + heuristics(env, curr_state)
        #print("curr_state: ", curr_state, ", new_limit: ", self.new_limit, " f_limit: ", f_limit, " new_f: ", new_f)
        if new_f > f_limit:
            self.new_limit = min(self.new_limit, new_f)
            return [], 0, 0

        if self.env.is_final_state(curr_state):  # check if final state
            costs[curr_state] -= 1
            return ucs_calc_results(self, env.get_initial_state(), curr_state, parents, actions, costs, len(expanded))

        for action, (new_state, cost, terminated) in env.succ(curr_state).items():  # loop over successors of new state
            # print("current state: ", curr_state, ", new state: ", new_state)
            # print("visited: ", visited[new_state], ", g + cost: ", g+cost, ", costs[new_state]: ", costs[new_state], ", terminated: ", terminated)
            if visited[new_state] == 0 and (env.is_final_state(new_state) or not terminated):
                # print("I'm inside the if statement")

                # update parameters
                path.append(new_state)
                actions[new_state] = action
                costs[new_state] = cost + g
                parents[new_state] = curr_state

                # recursion
                result = self.DFS_f(env, visited, expanded, parents, actions, costs, f_limit, new_state,
                                    path, g + cost)
                if result != ([], 0, 0):
                    return result

            elif visited[new_state] == 1 and (g + cost) < costs[new_state] and (
                    env.is_final_state(new_state) or not terminated):

                # update parameters
                path.append(new_state)
                actions[new_state] = action
                costs[new_state] = cost + g
                parents[new_state] = curr_state

                # recursion
                result = self.DFS_f(env, visited, expanded, parents, actions, costs, f_limit, new_state,
                                    path, g + cost)
                if result != ([], 0, 0):
                    return result
        return [], 0, 0
