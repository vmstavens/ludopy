#!/usr/bin/env python3
import argparse
import csv
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
from sklearn.metrics import ndcg_score
from word2number import w2n
import statsmodels.api as sm
import pylab as py
import scipy.stats as stats

def rank_chromosome(chromosome: np.array) -> np.array:
	action_names = np.array([action.name for action in ACTION])
	chromosome = np.array(chromosome)
	sorted_indices = np.argsort(chromosome)
	ranked_actions = list(zip(action_names[sorted_indices], chromosome[sorted_indices]))[::-1]
	return ranked_actions

from sklearn.metrics import ndcg_score
def ndcg(data1 : np.array, data2 : np.array) -> float:
	print("data1 = \n",data1)
	print("data2 = \n",data2)
	return ndcg_score([data1],[data2])


def str2ActionValues(action_list: list):
	def str2Action(str_action:str):
		actions = [action for action in ACTION]
		for a in actions:
			if a.name == str_action:
				return a
	action_values = []
	for a in action_list:
		action_values.append(str2Action(a).value)
	return action_values

def qqplot(data:np.array):
	stats.probplot(data, dist="norm", plot=py)
	py.show()


data_peter = np.load("final_run/_best_chromosome.npy")
data_victor = LudoPlayerGen(genome="data/elites/cr_0.75_se_tournament/test_name.csv").get_genome()[:,1]

data_peter = np.load('final_run/_evaluation.npy')[:1000,0]
np.savetxt("foo.csv", data_peter, delimiter=",")

print(data_peter[:1000].shape)

qqplot(data=data_peter)