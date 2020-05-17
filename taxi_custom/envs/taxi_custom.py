# -*- coding: utf-8 -*-

import sys
from contextlib import closing
from six import StringIO
from gym import utils
from gym.envs.toy_text import discrete
import numpy as np

MAP = [
    "+-------+",
    "|P: | : |",
    "| : : : |",
    "| : : : |",    
    "| | : | |",
    "| | : | |",
    "+-------+",
]


class TaxiCustomEnv(discrete.DiscreteEnv):
    """
    Sahil's and Miguel's Taxi Problem, 
    adapted from "Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition"
    by Tom Dietterich.
    Code adapted from Open AI Gym Taxi-v3 environment code.
    Description:
    When the episode starts, the taxi starts off at a random location and the passenger is in location P. The taxi drives to P, picks up the passenger, and drives to the passenger's destination D. It should also avoid a pothole H along the way. Once the passenger is dropped off, the episode ends.
    Observations: 
    There are 480 discrete states since there are 20 taxi positions (5x4 grid), 2 possible locations of the passenger (in or out of the taxi), 3 destination locations, and 4 pothole locations. 
    
    Passenger location:
    - 0: P (0,0)
    - 1: in taxi
    
    Destination D:
    - 0: (0,3)
    - 1: (4,0)
    - 2: (4,3)

    Pothole locations:
    - 0: (1,0)
    - 1: (1,3)
    - 2: (2,1)
    - 3: (2,3)
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    Rewards: 
    There is a reward of -1 for each action and an additional reward of +20 for delivering the passenger. There is a reward of -10 for executing actions "pickup" and "dropoff" illegally. There is also a reward of -8 for driving into the pothole.
    
    Rendering:
    - blue: passenger
    - magenta: destination
    - yellow: empty taxi
    - green: full taxi
    - letters (P, D, H): pick-up, destination and pothole locations
    
    state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination, pothole_location)
    """
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self):
        self.desc = np.asarray(MAP, dtype='c')

        self.dest_locs = dest_locs = [(0,3), (4,0), (4,3)]
        self.hole_locs = hole_locs = [(1,0), (1,1), (1,2), (1,3)]
        self.pass_locs = pass_locs = [(0,0)]

        num_states = 5*4*2*3*4 #rows X col X pass_loc X dest_loc X holes
        num_rows = 5
        num_columns = 4
        max_row = num_rows - 1
        max_col = num_columns - 1
        initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        P = {state: {action: []
                     for action in range(num_actions)} for state in range(num_states)}
        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(pass_locs)+1):  
                    for dest_idx in range(len(dest_locs)):
                        for hole_idx in range(len(hole_locs)):
                            state = self.encode(row, col, pass_idx, dest_idx, hole_idx)
                            if pass_idx == 0:
                                initial_state_distrib[state] += 1 #to ensure each initial state can occur with equal prob.
                            for action in range(num_actions):
                                # defaults
                                new_row, new_col, new_pass_idx = row, col, pass_idx
                                reward = -1 # default reward when there is no pickup/dropoff
                                done = False
                                taxi_loc = (row, col)

                                if action == 0:
                                    new_row = min(row + 1, max_row)
                                elif action == 1:
                                    new_row = max(row - 1, 0)
                                elif action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                    new_col = min(col + 1, max_col)
                                elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                    new_col = max(col - 1, 0)
                                if (new_row, new_col) == hole_locs[hole_idx]: # taxi drives into pothole
                                    reward = -8
                                elif action == 4:  # pickup
                                    if (pass_idx == 0 and taxi_loc == (0,0)):
                                        new_pass_idx = 1
                                    else: # passenger not at location
                                        reward = -10
                                elif action == 5:  # dropoff
                                    if (taxi_loc == dest_locs[dest_idx]) and pass_idx == 1:
                                        #new_pass_idx = dest_idx
                                        done = True
                                        reward = 20
                                    else: # dropoff at wrong location
                                        reward = -10
                                new_state = self.encode(
                                    new_row, new_col, new_pass_idx, dest_idx, hole_idx)
                                P[state][action].append(
                                    (1.0, new_state, reward, done))
        initial_state_distrib /= initial_state_distrib.sum()
        discrete.DiscreteEnv.__init__(
            self, num_states, num_actions, P, initial_state_distrib)

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx, hole_idx):
        # (5) 4, 2, 3, 4
        i = taxi_row
        i *= 4
        i += taxi_col
        i *= 2
        i += pass_loc
        i *= 3
        i += dest_idx
        i *= 4
        i += hole_idx
        
        return i

    def decode(self, i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 3)
        i = i // 3
        out.append(i % 2)
        i = i // 2
        out.append(i % 4)
        i = i // 4
        out.append(i)
        assert 0 <= i < 5
        return reversed(out)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        out = self.desc.copy().tolist()
        out = [[c.decode('utf-8') for c in line] for line in out]
        taxi_row, taxi_col, pass_idx, dest_idx, hole_idx = self.decode(self.s)
        
        hi, hj = self.hole_locs[hole_idx]
        out[1 + hi][2 * hj + 1] = 'H' # place pothole in grid
        
        di, dj = self.dest_locs[dest_idx]
        out[1 + di][2 * dj + 1] = 'D'

        def ul(x): return "_" if x == " " else x
        if pass_idx == 0:
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                out[1 + taxi_row][2 * taxi_col + 1], 'yellow', highlight=True)
            pi, pj = (0,0)
            out[1 + pi][2 * pj + 1] = utils.colorize(out[1 + pi][2 * pj + 1], 'blue', bold=True)
        else:  # passenger in taxi
            out[1 + taxi_row][2 * taxi_col + 1] = utils.colorize(
                ul(out[1 + taxi_row][2 * taxi_col + 1]), 'green', highlight=True)

        di, dj = self.dest_locs[dest_idx]
        out[1 + di][2 * dj + 1] = utils.colorize(out[1 + di][2 * dj + 1], 'magenta')

        outfile.write("\n".join(["".join(row) for row in out]) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
        else: outfile.write("\n")

        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
