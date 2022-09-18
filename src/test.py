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

parser = argparse.ArgumentParser()
parser.add_argument("-cr", "--crossover_rate", help="Crossover rate")
parser.add_argument("-se", "--selection_method", help="Selection method: \"tournament\" or \"roulette\"")

font = {'size'   : 11}

matplotlib.rc('font', **font)


now                           = datetime.now()
seed                          = now.strftime("%d-%m-%Y-%H-%M-%S")

args                          = parser.parse_args()
CROSSOVER_RATE                = float(args.crossover_rate)
SELECTION_METHOD              = args.selection_method
solutions                     = []
avg_avg_win_rates             = []
avg_win_rates                 = [0.0]
best_individual_in_generation = LudoPlayerGen()

CONFIG = f"_cr_{CROSSOVER_RATE}_se_{SELECTION_METHOD}"

file = open(f"data/avg_win_rate/data{CONFIG}_{seed}.csv", "a")

if not os.path.exists(f"data/models/test{CONFIG}_{seed}"):
	os.mkdir(f"data/models/test{CONFIG}_{seed}")

fig, ax = plt.subplots(2)

for s in range(NUMBER_OF_SOLUTIONS):
	solutions.append(LudoPlayerGen(genome=None, random=True, name="player_of_interest"))

for i in range(NUMBER_MAX_GENERATIONS):
 
	generation = []
	avg_win_rates = []
 
	print(f"Generation {i} started...")
	for sol in range(NUMBER_OF_SOLUTIONS):
		
		current_player_of_interest = solutions[sol]

		players = [
			current_player_of_interest,
			LudoPlayerRandom(),
			LudoPlayerRandom(),
			LudoPlayerRandom()
		]
	
		ludoGame = LudoGame(players)

		# evaluate the current player
		[fitness, avg_win_rate] = eval_player(
			players            = players,
			ludoGame           = ludoGame,
			player_of_interest = current_player_of_interest.name,
			num_of_runs        = NUMBER_OF_RUNS,
			individual         = sol)

		if current_player_of_interest.get_avg_win_rate() > best_individual_in_generation.get_avg_win_rate():
			best_individual_in_generation = current_player_of_interest

		generation.append(current_player_of_interest)

		avg_win_rates.append(avg_win_rate)

	best_individual_in_generation.save_to(f"data/models/test{CONFIG}_{seed}/gen_best_{i}.csv")
	# best_individual_in_generation.save_to(f"data/models/gen_best_{seed}.csv")
 
	avg_avg_win_rates.append(np.sum(avg_win_rates) / len(avg_win_rates))


	plt.clf()
	plt.xlabel("Generations #")
	plt.ylabel("Average winrate of individual-averages in a generation [%]")
	plt.ylim([0,1])
	plt.title("Avg winrate over generations - Genetic Algorithm")
	plt.plot(avg_avg_win_rates,label="Average of averages winrate")
	plt.legend()
	plt.savefig(f"img/img{CONFIG}_{seed}.pdf")
	# plt.show(block=False)
	# plt.pause(0.05)
 
	file.write(str(avg_avg_win_rates[len(avg_avg_win_rates) - 1]) + "\n") # log data to file
 
	random.shuffle(generation)
 
	better_solutions = natural_selection(generation,method=SELECTION_METHOD)
 
	next_generation = reproduce_generation(
		better_solutions, 
		crossover_rate   = CROSSOVER_RATE, 
		mutation_rate    = MUTATION_RATE, 
		crossover_method = CROSSOVER_METHOD)
 
	print("avg win rate: ",avg_avg_win_rates[len(avg_avg_win_rates) - 1])
	solutions = next_generation
