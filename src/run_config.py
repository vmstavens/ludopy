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
from state import ACTION
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-cp", "--chromosome_path", help="chromosome_path")
args = parser.parse_args()

now                           = datetime.now()
seed                          = now.strftime("%d-%m-%Y-%H-%M-%S")
avg_avg_win_rates             = []
avg_win_rates                 = [0.0]
NAME                          = args.chromosome_path.split("/")[2]
NUMBER_OF_EVAL_PLAYS          = 1000
print(NAME)

folder = f"data/elite_performance/{NAME}/"
file_name = f"{NAME}.csv"

if not os.path.exists(folder):
	os.makedirs(folder)
 
file = open(folder + file_name, "w")

print(args.chromosome_path)

player = LudoPlayerGen(genome=args.chromosome_path)

for item in range(NUMBER_OF_EVAL_PLAYS):

	players = [
		player,
		LudoPlayerRandom(),
		LudoPlayerRandom(),
		LudoPlayerRandom()
		]
	
	ludoGame = LudoGame(players)

	# evaluate the current player
	[fitness, avg_win_rate] = eval_player(
		players            = players,
		ludoGame           = ludoGame,
		player_of_interest = player.name,
		num_of_runs        = NUMBER_OF_RUNS,
		individual         = item)

	# 
	file.write(str(avg_win_rate) + "\n")