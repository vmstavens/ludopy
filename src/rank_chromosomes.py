#!/usr/bin/env python3
import argparse
from email import header
import sys
import time
import random
from traceback import print_tb
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
from game import LudoGame
from player_gen import LudoPlayerGen
from players import LudoPlayerDefensive, LudoPlayerRandom, LudoPlayerAggressive
from utils import crossover, eval_player, natural_selection, reproduce_generation
from config import NUMBER_MAX_GENERATIONS, NUMBER_OF_SOLUTIONS, NUMBER_OF_RUNS, MUTATION_RATE, CROSSOVER_RATE, CROSSOVER_METHOD, SELECTION_METHOD
import matplotlib
from state import ACTION
import pandas as pd

game_actions = [
		ACTION.MOVE_FROM_HOME,
		ACTION.MOVE_FROM_HOME_AND_KILL ,
		ACTION.MOVE ,
		ACTION.MOVE_ONTO_STAR ,
		ACTION.MOVE_ONTO_STAR_AND_DIE ,
		ACTION.MOVE_ONTO_STAR_AND_KILL ,
		ACTION.MOVE_ONTO_GLOBE ,
		ACTION.MOVE_ONTO_GLOBE_AND_DIE ,
		ACTION.MOVE_ONTO_ANOTHER_DIE ,
		ACTION.MOVE_ONTO_ANOTHER_KILL ,
		ACTION.MOVE_ONTO_VICTORY_ROAD ,
		ACTION.MOVE_ONTO_GOAL ,
		ACTION.NONE
]


def show_ranking(chromosome: np.array) -> np.array:
	action_names = np.array([action.name for action in game_actions])
	action_value = np.array([action.value for action in game_actions])
	print(len(chromosome))
	print(len(action_value))
	print(f"{action_value =}")
	print(f"{action_names =}")
	print(f"{chromosome = }")
	print(f"{chromosome.shape = }")
	chromosome = np.array(chromosome)
	# chromosome = np.array(chromosome)[:,1]
	sorted_indices = np.argsort(chromosome)
	ranked_actions = list(zip(action_names[sorted_indices], chromosome[sorted_indices]))[::-1]
	
	for rank, weighted_action in enumerate(ranked_actions):
		print(f"{rank, weighted_action = }")
	return ranked_actions

data_peter = np.load("final_run/_best_chromosome.npy")
data_peter = show_ranking(data_peter)

