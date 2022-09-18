#!/usr/bin/env python3

import random

from typing     import List, Tuple
from unittest import result

import numpy as np
from player_gen import LudoPlayerGen
from game       import LudoGame

def natural_selection(generation: List[LudoPlayerGen], method:str = "tournament") -> List[LudoPlayerGen]:
	"""A method for reducing the number of individuals in the generation given by a method

	Args:
		generation (List[LudoPlayerGen]): The generation we wish to reduce
		method (str, optional): The method by which the generation is reduced. Defaults to "tournament".

	Returns:
		List[LudoPlayerGen]: The reduced generation
	"""
	if method == "tournament":
		return tournament_selection(generation)
	elif method == "roulette":
		return roulette_selection(generation)


def mutate(player: LudoPlayerGen, mutation_rate:float = 0.01) -> LudoPlayerGen:
	"""Mutates the input player's genome with the mutation rate provided

	Args:
		player (LudoPlayerGen): The input player we wish to mutate
		mutation_rate (float, optional): The mutation rate. Defaults to 0.01.

	Returns:
		LudoPlayerGen: The mutated player
	"""
	new_genome = []
 
	genome = player.get_genome()
 
	for i in range(len(genome)):
		new_value = genome[i][1]
		if mutation_rate > random.random(): # if we should mutate
			new_value = genome[i][1] + random.uniform(-1, 1)
		new_genome.append((genome[i][0], new_value))
	player.set_genome(new_genome)
	return player


def crossover(player_1: LudoPlayerGen, player_2: LudoPlayerGen, crossover_rate: float = 0.3, method="probabilistic_cut") -> List[LudoPlayerGen]:
	"""Performs the crossover operation on two players genomes to produce 4 players 

	Args:
		player_1 (LudoPlayerGen): Player 1
		player_2 (LudoPlayerGen): Player 2
		crossover_rate (float, optional): The rate at which a crossover happens between two genomes. Defaults to 0.3.
		method (str, optional): The method applied to perform the crossover operations (\"probabilistic_cut\" or \"static_cut\"). Defaults to "probabilistic_cut".

	Returns:
		List[Tuple]: A list of 4 children produced by crossover
	"""
 
	child_genome_1 = LudoPlayerGen(genome=None)
	child_genome_2 = LudoPlayerGen(genome=None)
	child_genome_3 = LudoPlayerGen(genome=None)
	child_genome_4 = LudoPlayerGen(genome=None)

	children = [child_genome_1, child_genome_2, child_genome_3, child_genome_4]

	parents = [player_1.get_genome(), player_2.get_genome()]

	if method == "probabilistic_cut":
		for child in children:
			current_parent = parents[0]
			child_genome = []
			for i in range(len(player_1.get_genome())):
				if (random.random() < crossover_rate):  # perform a swap of parents
					current_parent = parents[0] if current_parent != parents[0] else parents[1]
				child_genome.append( (player_1.get_genome()[i][0], current_parent[i][1]) )
			child.set_genome(child_genome)
		return children

	elif method == "static_cut":
		for child in children:
			child_genome = []
   
			for i in range(len(player_1.get_genome())):
				if i < (crossover_rate * len(player_1.get_genome()) ):
					child_genome.append((player_1.get_genome()[i][0], parents[0][i][1]))
				else:
					child_genome.append((player_1.get_genome()[i][0], parents[1][i][1]))
			child.set_genome(child_genome)
		return children
	else:
		assert("Error: No invalid crossover mehtod passed, choose either: probabilistic_cut or static_cut")


def reproduce(player_1: LudoPlayerGen, player_2: LudoPlayerGen, crossover_rate: float, mutation_rate: float, crossover_method: str = "probabilistic_cut") -> List[LudoPlayerGen]:
	"""Produces offspring 4 players based on the 2 input players

	Args:
		player_1 (LudoPlayerGen): Parent 1
		player_2 (LudoPlayerGen): Parent 2
		crossover_rate (float): The crossover rate
		mutation_rate (float): The Mutation rate
		crossover_method (str): The Method by which crossover is applied. Defaults to \"probabilistic_cut\".

	Returns:
		List[LudoPlayerGen]: 4 children
	"""
	# crossover
	children = crossover(
		player_1       = player_1, 
		player_2       = player_2,
		crossover_rate = crossover_rate,
		method         = crossover_method)

	# mutate
	mutated_players = []
	for child in children:
		mut_child = mutate(player=child,mutation_rate=mutation_rate)
		mutated_players.append(mutate(player=child,mutation_rate=mutation_rate))

	return mutated_players

def reproduce_generation(generation: List[LudoPlayerGen], crossover_rate: float = 0.3, mutation_rate: float = 0.01, crossover_method: str = "probabilistic_cut") -> List[LudoPlayerGen]:
	"""Reproduces and entire generation of genetic algorithm players

	Args:
		generation (List[LudoPlayerGen]): The input generation and their fitness scores
		crossover_rate (float, optional): The crossover rate used for reproduction. Defaults to 0.3.
		mutation_rate (float, optional): The mutation rate used for reproduction. Defaults to 0.01.
  		crossover_method (str): The Method by which crossover is applied. Defaults to \"probabilistic_cut\".

	Returns:
		List[LudoPlayerGen]: The produced generation from reproduction
	"""
	new_generation = []
 
	males          = generation[:][::2]
	females        = generation[:][1::2]
	for i in range(len(males)):
		new_generation += reproduce(males[i],females[i],crossover_rate,mutation_rate,crossover_method)
	return new_generation


def eval_player(players: List[LudoPlayerGen], ludoGame: LudoGame, player_of_interest: str = "genetic_algorithm", num_of_runs: int = 100, individual:int = 0) -> Tuple[float, float]:
	"""Evaluate the player of interest vs. the other players in \"players\" from the game ludoGame

	Args:
		players (List[LudoPlayerGen]): The list of 4 players containing both the player of interest and its opponents
		ludoGame (LudoGame): The ludogame the players are going to play in
		player_of_interest (str, optional): The player name of the player of interest. Defaults to "genetic_algorithm".
		num_of_runs (int, optional): The number of runs to evaluate the player. Defaults to 100.

	Returns:
		Tuple[float, float]: The average winrate and fitness score of the player of interest
	"""

	# parameters to keep track of
	#  + win/loss
	#  + number of deaths

	# shuffle players and start game
	random.shuffle(players)

	avg_win_rate = 0.0

	player_gen_id = get_player_id(players,player_of_interest)
	
	# play NUMBER_OF_RUNS number of runs to get an average performance
	for run in range(num_of_runs):
		print(f"\tIndividual {individual + 1}; Run {run + 1}/{num_of_runs}...",end="\r")
		ludoGame = LudoGame(players)
		winner = ludoGame.play_full_game()

		if winner == player_gen_id:
			avg_win_rate += 1
 
	print("")
	# genetic player
	player_gen = players[player_gen_id]
 
	avg_win_rate = avg_win_rate / float(num_of_runs)
 
	# set genetic player win rate
	player_gen.set_avg_win_rate(avg_win_rate)
 
	player_gen.calc_fitness(ludoGame.state,player_gen_id)

	return [player_gen.get_fitness(), player_gen.get_avg_win_rate()]

def tournament_selection(generation: List[LudoPlayerGen]) -> List[LudoPlayerGen]:
	best_selection = []
 
	competitors_1 = generation[::2]
	competitors_2 = generation[1::2]
 
	for i in range(len(competitors_1)):
		if competitors_1[i].get_fitness() >= competitors_2[i].get_fitness():
			best_selection.append(competitors_1[i])
		else:
			best_selection.append(competitors_2[i])
	return best_selection


def roulette_selection(generation: List[LudoPlayerGen]) -> List[LudoPlayerGen]:
	# Computes the totallity of the gen fitness
	gen_fitness = sum([chromosome.get_fitness() for chromosome in generation])
	# Computes for each chromosome the probability 
	chromosome_probabilities = [chromosome.get_fitness()/gen_fitness for chromosome in generation]
	# Selects one chromosome based on the computed probabilities
	if np.count_nonzero(chromosome_probabilities == 0.0) < int(len(generation)*0.5):
		return generation[0:int(0.5*len(generation))]
	return np.random.choice(generation,size=int(len(generation)*0.5), p=chromosome_probabilities,replace=False)

def get_player_id(players:List, player_of_interest:str)->int:
	for i in range(len(players)):
		if players[i].name == player_of_interest:
			return i
