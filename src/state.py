#!/usr/bin/env python3
import numpy as np
import pdb
from aenum import Enum, IntEnum, NoAlias, auto

# star_jump, is_on_globe, steps_taken, token_vulnerability, will_send_self_home, will_send_opponent_home, will_send_self_onto_goal, will_send_self_onto_victory_road, will_win_game, will_move_from_home, is_home, is_on_common_path, is_on_victory_road, is_in_goal, can_kill
from helpers import *

class ACTION(IntEnum):

	MOVE_FROM_HOME = 0
	MOVE_FROM_HOME_AND_KILL = auto()
	MOVE = auto()
	MOVE_ONTO_STAR = auto()
	MOVE_ONTO_STAR_AND_DIE = auto()
	MOVE_ONTO_STAR_AND_KILL = auto()
	MOVE_ONTO_GLOBE = auto()
	MOVE_ONTO_GLOBE_AND_DIE = auto()
	MOVE_ONTO_ANOTHER_DIE = auto()
	MOVE_ONTO_ANOTHER_KILL = auto()
	MOVE_ONTO_VICTORY_ROAD = auto()
	MOVE_ONTO_GOAL = auto()
	NONE = auto()

class STATE(IntEnum):

	HOME = 0
	HOME_CAN_KILL = auto()
	GLOBE = auto()
	GLOBE_CAN_KILL = auto()
	GLOBE_IN_DANGER = auto()
	GLOBE_IN_DANGER_CAN_KILL = auto()
	# STAR                            = auto()
	# STAR_CAN_KILL                   = auto()
	# STAR_IN_DANGER                  = auto()
	COMMON_PATH = auto()  # same for STAR
	COMMON_PATH_CAN_KILL = auto()
	COMMON_PATH_IN_DANGER = auto()
	COMMON_PATH_IN_DANGER_CAN_KILL = auto()
	COMMON_PATH_WITH_BUDDY = auto()
	COMMON_PATH_WITH_BUDDY_CAN_KILL = auto()
	VICTORY_ROAD = auto()
	GOAL = auto()

class REWARD(Enum, settings=NoAlias):

	MOVE = 5  # proportional to steps taken
	MOVE_FROM_HOME = 5  # + bonus if kill
	DIE = -50
	KILL = 5  # + extra proportional to steps taken
	GET_ONTO_VICTORY_ROAD = 50
	GET_IN_GOAL = 100
	WIN = 1000
	MOVE_TO_SAFETY = 25
	MOVE_TO_DANGER = -25

class LudoState:
	def __init__(self, state=None, empty=False):
		if state is not None:
			self.state = state
		else:
			# 4 players, 4 tokens per player
			self.state = np.empty((4, 4), dtype=np.int)
			if not empty:
				self.state.fill(-1)

	def copy(self):
		return LudoState(self.state.copy())

	def __getitem__(self, item):
		return self.state[item]

	def __setitem__(self, key, value):
		self.state[key] = value

	def __iter__(self):
		return self.state.__iter__()

	def get_state(self, token_id, player_id=0):

		token_pos = self.state[player_id, token_id]

		if token_pos == -1:
			return STATE.HOME

		elif token_pos == 99:
			return STATE.GOAL

		elif is_on_globe(token_pos):
			return STATE.GLOBE

		elif (token_pos > 51) and (token_pos < 99):
			return STATE.VICTORY_ROAD

		elif (token_pos > 0) and (token_pos < 53):
			return STATE.COMMON_PATH

	def get_state_advanced(self, token_id, player_id=0):

		token_pos = self.state[player_id, token_id]
		opponents = self.state[1:]

		if is_home(token_pos):
			return STATE.HOME_CAN_KILL if np.sum(opponents == 1) > 0 else STATE.HOME

		if is_in_goal(token_pos):
			return STATE.GOAL

		in_danger = token_vulnerability(self.state, token_id) >= 1
		can_kill = token_can_kill(self.state, token_pos)

		if is_on_globe(token_pos):
			if in_danger and can_kill:
				return STATE.GLOBE_IN_DANGER_CAN_KILL
			if in_danger:
				return STATE.GLOBE_IN_DANGER
			if can_kill:
				return STATE.GLOBE_CAN_KILL
			return STATE.GLOBE

		if is_on_victory_road(token_pos):
			return STATE.VICTORY_ROAD

		num_buddies = 0 if token_pos == - \
			1 else (np.sum(self.state[0] == token_pos) - 1)

		if is_on_common_path(token_pos) and num_buddies >= 1:
			return STATE.COMMON_PATH_WITH_BUDDY_CAN_KILL if can_kill else STATE.COMMON_PATH_WITH_BUDDY

		if is_on_common_path(token_pos):
			if in_danger and can_kill:
				return STATE.COMMON_PATH_IN_DANGER_CAN_KILL
			if in_danger:
				return STATE.COMMON_PATH_IN_DANGER
			if can_kill:
				return STATE.COMMON_PATH_CAN_KILL
			return STATE.COMMON_PATH

	def get_reward(self, next_state, bonus=False):
		""" return reward tuple (name, value) for player 0 (for relative states) """

		LONGEST_STEP = 13

		# maybe cancel bonus if only one token is movable
		if bonus:
			# change in vulnerabilty (prev - new = score, >0 = good, <0 = bad, =0 = neutral)
			delta_vulnerability = sum([token_vulnerability(self.state, i) for i in range(
				4)]) - sum([token_vulnerability(next_state, i) for i in range(4)])
			bonus = REWARD.MOVE_TO_SAFETY.value if delta_vulnerability > 0 else (
				REWARD.MOVE_TO_DANGER.value if delta_vulnerability < 0 else 0)
		else:
			bonus = 0

		if will_send_self_home(self, next_state):
			return REWARD.DIE, REWARD.DIE.value

		elif will_move_from_home(self, next_state):
			return REWARD.MOVE_FROM_HOME, REWARD.MOVE_FROM_HOME.value + (REWARD.KILL.value if will_send_opponent_home(self, next_state) else 0) + bonus

		elif will_send_opponent_home(self, next_state):
			return REWARD.KILL, REWARD.KILL.value + steps_taken(self, next_state)/LONGEST_STEP * REWARD.MOVE.value + bonus

		elif will_send_self_onto_victory_road(self, next_state):
			return REWARD.GET_ONTO_VICTORY_ROAD, REWARD.GET_ONTO_VICTORY_ROAD.value + bonus

		elif will_win_game(next_state):
			return REWARD.WIN, REWARD.WIN.value + bonus

		elif will_send_self_onto_goal(self, next_state):
			return REWARD.GET_IN_GOAL, REWARD.GET_IN_GOAL.value + bonus

		else:
			return REWARD.MOVE, steps_taken(self, next_state)/LONGEST_STEP * REWARD.MOVE.value + bonus

	@staticmethod
	def get_tokens_relative_to_player(tokens, player_id):
		if player_id == 0:
			return tokens

		rel_tokens = []
		for token_id, token_pos in enumerate(tokens):
			if token_pos == -1 or token_pos == 99:  # start and end pos are independent of player id
				rel_tokens.append(token_pos)
			elif token_pos < 52:  # in common area
				rel_tokens.append((token_pos - player_id * 13) % 52)
			else:  # in end area, 52 <= x < 52 + 20
				rel_tokens.append(((token_pos - 52 - player_id * 5) % 20) + 52)
		return rel_tokens

	def get_state_relative_to_player(self, rel_player_id, keep_player_order=False):
		if rel_player_id == 0:
			return self.copy()

		rel = LudoState(empty=True)
		new_player_ids = list(range(4)) if keep_player_order else [
                    (x - rel_player_id) % 4 for x in range(4)]

		for player_id, player_tokens in enumerate(self):
			new_player_id = new_player_ids[player_id]
			rel[new_player_id] = self.get_tokens_relative_to_player(
				player_tokens, rel_player_id)

		return rel

	def move_token(self, token_id, dice_roll):
		"""
		compute the move for token of player 0 in current state
		return resulting tuple of (new State(), ACTION() taken)
		"""

		cur_pos = self[0][token_id]
		# current_state = self.get_state(token_id)

		# if token in goal, no actions possible
		if cur_pos == 99:
			return False, ACTION.NONE

		new_state = self.copy()
		player = new_state[0]
		opponents = new_state[1:]

		# move from home if token in home and a 6 is rolled
		if cur_pos == -1:
			if dice_roll != 6:
				return False, ACTION.NONE

			player[token_id] = 1
			kill = np.sum(opponents == 1) > 0
			opponents[opponents == 1] = -1  # kill

			return new_state, ACTION.MOVE_FROM_HOME_AND_KILL if kill else ACTION.MOVE_FROM_HOME

		target_pos = cur_pos + dice_roll

		# common area move
		if target_pos < 52:

			occupants = opponents == target_pos
			occupant_count = np.sum(occupants)

			# occupied by multiple other tokens
			if occupant_count > 1:
				player[token_id] = -1  # sends self home
				return new_state, ACTION.MOVE_ONTO_ANOTHER_DIE

			# globe
			if (occupant_count == 1 and is_on_globe(target_pos)):
				player[token_id] = -1  # sends self home
				return new_state, ACTION.MOVE_ONTO_GLOBE_AND_DIE
			elif (is_on_globe(target_pos)):
				player[token_id] = target_pos
				return new_state, ACTION.MOVE_ONTO_GLOBE

			# star
			if (star_jump_length := star_jump(target_pos)):

				kill = False
				# check for kills on current star
				if occupant_count == 1:
					opponents[occupants] = -1
					kill = True

				# last star -> send directly to goal
				if target_pos == 51:
					player[token_id] = 99
					return new_state, ACTION.MOVE_ONTO_GOAL

				else:

					target_pos = target_pos + star_jump_length

					occupants = opponents == target_pos
					occupant_count = np.sum(occupants)

					# multiple opponent tokens on ending star
					if occupant_count > 1:
						player[token_id] = -1  # sends self home
						return new_state, ACTION.MOVE_ONTO_STAR_AND_DIE

					# one token on opponent star, kill it
					elif (occupant_count == 1):
						opponents[occupants] = -1
						kill = True

					# perform move
					player[token_id] = target_pos
					return new_state, (ACTION.MOVE_ONTO_STAR_AND_KILL if kill else ACTION.MOVE_ONTO_STAR)

			# normal move
			player[token_id] = target_pos

			if occupant_count == 1:
				opponents[occupants] = -1
				return new_state, ACTION.MOVE_ONTO_ANOTHER_KILL
			else:
				return new_state, ACTION.MOVE

		# victory road move
		if target_pos == 57:  # token reached goal
			player[token_id] = 99
			return new_state, ACTION.MOVE_ONTO_GOAL

		elif target_pos < 57:  # no goal bounce
			player[token_id] = target_pos
			return new_state, (ACTION.MOVE_ONTO_VICTORY_ROAD if cur_pos < 52 else ACTION.MOVE)

		else:  # bounce back from goal pos
			player[token_id] = 57 - (target_pos - 57)
		return new_state, ACTION.MOVE

	def get_winner(self):

		for player_id in range(4):
			if np.all(self[player_id] == 99):
				return player_id

		return -1


class LudoStateFull:
	def __init__(self, state, roll, next_states):
		self.state = state
		self.roll = roll
		self.next_states = next_states
