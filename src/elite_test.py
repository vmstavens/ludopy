#!/usr/bin/env python3
import argparse
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
import pandas as pd
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-cr", "--crossover_rate", help="Crossover rate")
parser.add_argument("-se", "--selection_method", help="Selection method: \"tournament\" or \"roulette\"")
args = parser.parse_args()


models_path = "data/models/"
models_type = [
    "test_cr_0.5_se_roulette/",
    "test_cr_0.25_se_roulette/",
    "test_cr_0.75_se_roulette/",
    "test_cr_0.5_se_tournament/",
    "test_cr_0.25_se_tournament/",
    "test_cr_0.75_se_tournament/"]

DEBUG = [False,True][0]

if DEBUG:
	NUM_OF_PLAYERS = 2
	NUM_OF_ROUNDS = 3
else:
	NUM_OF_PLAYERS = 150
	NUM_OF_ROUNDS = 50

players = []
solutions = []

CONFIG = f"cr_{args.crossover_rate}_se_{args.selection_method}"

SAVE_PATH = f"data/elites/{CONFIG}/"

FILE_NAME = "test_name.csv"

if not os.path.exists(SAVE_PATH):
	os.makedirs(SAVE_PATH)

# "test_cr_0.5_se_roulette/"
for i in range(NUM_OF_PLAYERS):
	chromosome = pd.read_csv(models_path + f"{CONFIG}/" + f"gen_best_{i}.csv",header=None,index_col=False).to_numpy()
	solutions.append(LudoPlayerGen(genome=chromosome,random=False))

best_gen_player = LudoPlayerGen()

for i in range(NUM_OF_PLAYERS):
	players = [
			solutions[i],
			LudoPlayerRandom(),
			LudoPlayerRandom(),
			LudoPlayerRandom()
		]
	ludoGame = LudoGame(players)
	[fitness, avg_win_rate] = eval_player(
				players            = players,
				ludoGame           = ludoGame,
				player_of_interest = solutions[i].name,
				num_of_runs        = NUM_OF_ROUNDS,
				individual         = i)

	if best_gen_player.get_avg_win_rate() < solutions[i].get_avg_win_rate():
		best_gen_player = solutions[i]

best_gen_player.save_to(SAVE_PATH + FILE_NAME)

print("best win rate => ",best_gen_player.get_avg_win_rate())