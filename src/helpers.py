from audioop import cross
import numpy as np
import collections
import random

from typing import List, Tuple

import game

def randargmax(x, **kw):
	""" a random tie-breaking argmax"""
	# https://stackoverflow.com/questions/42071597/numpy-argmax-random-tie-breaking/42071648
	return np.argmax(np.random.random(x.shape) * (x==x.max()), **kw)
	
def running_avg(new_val, size=10):
	
	if not hasattr(running_avg, "buffer"):
		running_avg.mean = 0.
		running_avg.buffer = collections.deque([0] * size, maxlen=size)
		
	last_val = running_avg.buffer.popleft()
	running_avg.buffer.append(new_val)
	running_avg.mean += 1/size * (new_val - last_val)

	return running_avg.mean

def star_jump(token_pos):
	if token_pos == -1 or token_pos > 51:
		return 0
	if token_pos % 13 == 6:
		return 6
	if token_pos % 13 == 12:
		return 7
	return 0

def is_home(token_pos):
	return token_pos == -1
	
def is_in_goal(token_pos):
	return token_pos == 99
	
def is_on_globe(token_pos):
	if token_pos == -1 or token_pos > 51:
		return False
	if token_pos % 13 == 1:
		return True
	if token_pos % 13 == 9:
		return True
	return False

def is_on_common_path(token_pos):
	return (token_pos > 0) and (token_pos < 53)
	
def is_on_victory_road(token_pos):
	return (token_pos > 51) and (token_pos < 99)

def valid_dice_roll(n):
	return 1 <= n <= 6

def will_send_self_home(state, next_state):
	return np.sum(state[0] == -1) < np.sum(next_state[0] == -1)

def will_send_opponent_home(state, next_state):
	return np.sum(state[1:] == -1) < np.sum(next_state[1:] == -1)
	
def will_send_self_onto_goal(state, next_state):
	return np.sum(state[0] == 99) < np.sum(next_state[0] == 99)

def will_send_self_onto_victory_road(state, next_state):
	return np.sum((state[0] >= 52) & (state[0] <= 56)) < np.sum((next_state[0] >= 52) & (next_state[0] <= 56))

def will_win_game(next_state):
	return np.all(next_state[0] == 99)
	
def will_move_from_home(state, next_state):
	return np.sum(state[0] == -1) > np.sum(next_state[0] == -1)
	
def steps_taken(state, next_state):
	return np.sum(next_state[0] - state[0])
	
def token_can_kill(state, token_pos):	
	return np.any((state[1:] > token_pos) & (state[1:] < (token_pos + 6)) & ~( (state[1:] % 13 == 1) | (state[1:] % 13 == 9) ))

def token_vulnerability(state, token_id):
	""" returns an approximation of the amount (n) of opponent dice rolls that can send the token home """
	player = state[0]
	token = player[token_id]

	if token == -1 or token == 1 or token > 51:  # in home, start or end positions
		return 0
	if token % 13 == 1 and np.sum(state[token // 13] == -1) == 0:  # on globe outside empty home
		return 0
	if token % 13 != 1 and np.sum(player == token) > 1 or token % 13 == 9:  # blockade or globe
		return 0

	n = 0

	if token % 13 == 1:  # on opponent start pos
		n += 1

	star = star_jump(token)
	if star > 0:
		star = 6 if star == 7 else 7

	for opponent_id in range(1, 4):
		opponent = state[opponent_id]
		for opp_token in set(opponent):
			if opp_token == -1 or opp_token > 51:
				continue
			req_dice_roll = (token - opp_token) % 52
			rel_opp_token = (opp_token - opponent_id * 13) % 52
			would_enter_end_zone = rel_opp_token + req_dice_roll > 51
			if not would_enter_end_zone and 1 <= req_dice_roll <= 6:
				n += 1
			if star > 0:
				req_dice_roll = (token - opp_token - star) % 52
				would_enter_end_zone = rel_opp_token + req_dice_roll + star > 51
				if not would_enter_end_zone and 1 <= req_dice_roll <= 6:
					n += 1
	return n


