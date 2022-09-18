from argparse import _get_action_name
from ast import List
from multiprocessing.dummy import Array
from sre_parse import State
from typing import Tuple, List
from state import ACTION, REWARD, STATE, LudoState
from config import HOME_POSITION, TOKENS_ON_GOAL_SCALING
import csv
# from helpers import randargmax, will_send_opponent_home, will_send_self_home

import random
import logging
import numpy as np

class LudoPlayerGen:
	""" player trained with Genetic Algorithm """

	def __init__(self,genome=None,random=False ,**kwargs):
		
		self.actions = [
      				ACTION.MOVE_FROM_HOME,
					ACTION.MOVE,
					ACTION.MOVE_FROM_HOME_AND_KILL,
					ACTION.MOVE_ONTO_GLOBE,
					ACTION.MOVE_ONTO_GLOBE_AND_DIE,
					ACTION.MOVE_ONTO_VICTORY_ROAD,
					ACTION.MOVE_ONTO_STAR,
					ACTION.MOVE_ONTO_STAR_AND_DIE,
					ACTION.MOVE_ONTO_STAR_AND_KILL,
					ACTION.MOVE_ONTO_ANOTHER_DIE,
					ACTION.MOVE_ONTO_ANOTHER_KILL,
					ACTION.MOVE_ONTO_GOAL 
		]
  
		self.action_values = [
			0.5, # ACTION.MOVE_FROM_HOME
			0.5, # ACTION.MOVE
			0.5, # ACTION.MOVE_FROM_HOME_AND_KILL
			0.5, # ACTION.MOVE_ONTO_GLOBE
			0.5, # ACTION.MOVE_ONTO_GLOBE_AND_DIE
			0.5, # ACTION.MOVE_ONTO_VICTORY_ROAD
			0.5, # ACTION.MOVE_ONTO_STAR
			0.5, # ACTION.MOVE_ONTO_STAR_AND_DIE
			0.5, # ACTION.MOVE_ONTO_STAR_AND_KILL
			0.5, # ACTION.MOVE_ONTO_ANOTHER_DIE
			0.5, # ACTION.MOVE_ONTO_ANOTHER_KILL
			0.5, # ACTION.MOVE_ONTO_GOAL
		]
     
		self.name = kwargs['name'] if 'name' in kwargs else "genetic_algorithm"
		self.move_from_home          = kwargs['move_from_home'] if 'move_from_home' in kwargs else                   (self.actions[0],  self.action_values[0])
		self.move                    = kwargs['move_casual'] if 'move_casual' in kwargs else                         (self.actions[1],  self.action_values[1])
		self.move_from_home_and_kill = kwargs['move_from_home_and_kill'] if 'move_from_home_and_kill' in kwargs else (self.actions[2],  self.action_values[2])
		self.move_onto_globe         = kwargs['move_onto_globe'] if 'move_onto_globe' in kwargs else                 (self.actions[3],  self.action_values[3])
		self.move_onto_globe_and_die = kwargs['move_onto_globe_and_die'] if 'move_onto_globe_and_die' in kwargs else (self.actions[4],  self.action_values[4])
		self.move_onto_victory_road  = kwargs['move_onto_victory_road'] if 'move_onto_victory_road' in kwargs else   (self.actions[5],  self.action_values[5])
		self.move_onto_star          = kwargs['move_onto_star'] if 'move_onto_star' in kwargs else                   (self.actions[6],  self.action_values[6])
		self.move_onto_star_and_die  = kwargs['move_onto_star_and_die'] if 'move_onto_star_and_die' in kwargs else   (self.actions[7],  self.action_values[7])
		self.move_onto_star_and_kill = kwargs['move_onto_star_and_kill'] if 'move_onto_star_and_kill' in kwargs else (self.actions[8],  self.action_values[8])
		self.move_onto_another_die   = kwargs['move_onto_another_die'] if 'move_onto_another_die' in kwargs else     (self.actions[9],  self.action_values[9])
		self.move_onto_another_kill  = kwargs['move_onto_another_kill'] if 'move_onto_another_kill' in kwargs else   (self.actions[10], self.action_values[10])
		self.move_onto_goal          = kwargs['move_onto_goal'] if 'move_onto_goal' in kwargs else                   (self.actions[11], self.action_values[11])

		self.genome                      = [None] * (len(ACTION) - 1)
		self.avg_fitness_score           = 0.0
		self.avg_num_of_kills            = 0.0
		self.avg_win_rate                = 0.0
		self.avg_num_of_tokens_on_goal   = 0.0
   
		if isinstance(genome, str):  # from path
			self.genome_path = genome
			self.genome = np.loadtxt(self.genome_path, delimiter=",")
   
		elif isinstance(genome, np.ndarray):  # shared
			self.genome = genome
   
		elif random:  # randomly generated player action values
			self.genome = self.random_genome()
   
		# initialize empty
		elif genome is None and random == False:  # empty
			gene_list = []
			for i in range(len(self.genome)):
				gene_list.append((self.actions[i], 0))
			self.set_genome(gene_list)
   
		else: # default case
			self.genome = [
				self.move_from_home,
				self.move,
				self.move_from_home_and_kill,
				self.move_onto_globe,
				self.move_onto_globe_and_die,
				self.move_onto_victory_road,
				self.move_onto_star,
				self.move_onto_star_and_die,
				self.move_onto_star_and_kill,
				self.move_onto_another_die,
				self.move_onto_another_kill,
				self.move_onto_goal
			]

	def random_genome(self) -> List[Tuple[ACTION,float]]:
		"""Generates a random genome, which is a List[Tuple[ACTION,float]]

		Returns:
			List[Tuple[ACTION,float]]: A random genome
		"""
		genome = []
		for i in range(len(self.actions)):
			genome.append( (self.actions[i] ,random.random()) )   
		return genome

	def get_avg_win_rate(self) -> float:
		"""Return the instance's average winrate

		Returns:
			float: average win rate
		"""
		return self.avg_win_rate

	def set_avg_win_rate(self, new_avg_win_rate) -> None:
		"""Sets the instance's average win rate

		Args:
			new_avg_win_rate (float, optional): The new average win rate. Defaults to 0.0.
		"""
		self.avg_win_rate = new_avg_win_rate

	def get_genome(self) -> List[Tuple[ACTION, float]]:
		"""Returns the instance's genome

		Returns:
			List[Tuple[ACTION,float]]: The instance's genome
		"""
		return self.genome

	def set_genome(self,genome:List[Tuple[ACTION,float]]) -> None:
		"""Sets the instance's genome

		Args:
			genome (List[Tuple[ACTION,float]]): The genome we wish to set the instance's genome as
		"""
		self.genome = genome

	def calc_fitness(self,state:State,player_gen_id:int) -> None:
		"""Calculates the instances fitness score and sets the corresponding member

		Returns:
			Nonce: 
		"""
		self.fitness_score = self.fitness(state=state,player_gen_id=player_gen_id)

	def get_fitness(self) -> float:
		"""Returns the fitness score

		Returns:
			float: Fitness score
		"""
		return self.fitness_score

	def set_fitness_score(self,fitness_score: float = 0.0):
		"""Sets the fitness score of this instance

		Args:
			fitness_score (float): The new fitness score for this instance
		"""
		self.fitness_score = fitness_score

	def get_num_of_kills(self):
		"""Returns the number of kills this instance has gotten

		Returns:
			int: Number of kills
		"""
		return self.num_of_kills

	def is_action_a_kill(self, action: ACTION) -> bool:
		"""Determines if the input action is a killing action, meaning does this action kill an opponent token

		Args:
			action (ACTION): The action of interest

		Returns:
			bool: True if the action is a kill, otherwise False
		"""
		if (action == ACTION.MOVE_ONTO_ANOTHER_KILL) or \
			(action == ACTION.MOVE_FROM_HOME_AND_KILL) or \
			(action == ACTION.MOVE_ONTO_STAR_AND_KILL):
			return True
		else:
			return False

	def is_action_a_home_run(self,action:ACTION) -> bool:
		"""Determines if the input action is an action which causes the token to move onto goal

		Args:
			action (ACTION): action

		Returns:
			bool: true if action results in goal
		"""
		if (action == ACTION.MOVE_ONTO_GOAL):
			return True
		else:
			return False

	def has_won(self,player_id:int, state:State) -> bool:
		sum_of_token_1_moves = state.state[player_id, 0]
		sum_of_token_2_moves = state.state[player_id, 1]
		sum_of_token_3_moves = state.state[player_id, 2]
		sum_of_token_4_moves = state.state[player_id, 3]
  
		if (sum_of_token_1_moves == HOME_POSITION) and \
      		(sum_of_token_2_moves == HOME_POSITION) and \
			(sum_of_token_3_moves == HOME_POSITION) and \
			(sum_of_token_4_moves == HOME_POSITION):
			return True
		else:
			return False

	def play(self, state: LudoState, dice_roll:int, next_states_actions : List) -> int:

		# given the current state, 
		# determine which token is in a favorable state 
  
		best_token_action = -1
		best_token = -1
		for i in range(len(next_states_actions)):
			action = next_states_actions[i][1]
			action_value = self.get_action_value(action)

			# does the action cause the token to move to goal
			if self.is_action_a_home_run(action):
				self.avg_num_of_tokens_on_goal += 1

			if self.is_action_a_kill(action):
				self.avg_num_of_kills += 1
   
			# if the current state action pair is (False, ACTION.NONE)
			if not self.can_be_played(next_states_actions[i]) or action_value == None:
				continue

			if action_value >= best_token_action:
				best_token = i
				best_token_action = action_value
		return best_token

	def get_action_value(self,action: ACTION):
		for gen in self.genome:
			# search for action in genome and return the genome value for said action
			if action == gen[0]:
				return gen[1]
			else:
				assert "No valid action found in get_action_value(self,action: ACTION)"

	def can_be_played(self, next_state_action):
		if ((next_state_action[0] != False) or \
			(next_state_action[1] != ACTION.NONE)):
			return True
		else:
			return False

	def fitness(self, state: LudoState, player_gen_id: int) -> float:  
		# f = wr * (ank + anth)
		# f = 0.25 * ( 4 + 2 )
		self.fitness_score = self.avg_win_rate * (self.avg_num_of_kills + self.avg_num_of_tokens_on_goal * TOKENS_ON_GOAL_SCALING)
		return self.fitness_score

	def save_to(self,save_path:str = "data/models/test.csv"):
		print("save_path = ",save_path)
		with open(save_path, 'w') as f:
			writer = csv.writer(f)
			for gene in self.genome:
				writer.writerow([gene[0], gene[1]])