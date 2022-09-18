from cProfile import label
from datetime import datetime
from turtle import shape
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from requests import head
import matplotlib

# DATA_PATH = "data/models/test_24-04-2022-10-33-24/"
# DATA_PATH = "data/models/test_23-04-2022-19-04-53/"

font = {'size'   : 12}

matplotlib.rc('font', **font)

DP_CR_25_ROU = "data/avg_win_rate/data_cr_0.25_se_roulette.csv"
DP_CR_50_ROU = "data/avg_win_rate/data_cr_0.5_se_roulette.csv"
DP_CR_75_ROU = "data/avg_win_rate/data_cr_0.75_se_roulette.csv"
DP_CR_25_TOU = "data/avg_win_rate/data_cr_0.25_se_tournament.csv"
DP_CR_50_TOU = "data/avg_win_rate/data_cr_0.5_se_tournament.csv"
DP_CR_75_TOU = "data/avg_win_rate/data_cr_0.75_se_tournament.csv"

file_paths = [
	DP_CR_25_TOU,
	DP_CR_50_TOU,
	DP_CR_75_TOU,
    DP_CR_25_ROU,
	DP_CR_50_ROU,
	DP_CR_75_ROU]

now                           = datetime.now()
seed                          = now.strftime("%d-%m-%Y-%H-%M-%S")

# NUM_OF_ACTIONS = 12

# _, _, files = next(os.walk(DATA_PATH))
# num_of_files = len(files)

# data = []

# for i in range(num_of_files):
# 	df = pd.read_csv(DATA_PATH + f"gen_best_{i}.csv",header=None).iloc[:,1].values.T
# 	data.append(df)


avg_avg_data = []

for i in range(len(file_paths)):
	df = pd.read_csv(file_paths[i],header=None)[0].values
	avg_avg_data.append(df)

avg_avg_data = np.array(avg_avg_data)

labels = ["C1",
			"C2",
			"C3",
			"C4",
			"C5",
			"C6"]
# labels = ["crossover rate 0.25, roulette selection",
# 			"crossover rate 0.50, roulette selection",
# 			"crossover rate 0.75, roulette selection",
# 			"crossover rate 0.25, tournament selection",
# 			"crossover rate 0.50, tournament selection",
# 			"crossover rate 0.75, tournament selection"]

# df = pd.DataFrame(avg_avg_data)
# # df = pd.DataFrame(avg_avg_data,columns=["1","1","1","1","1","1"])
for i in range(len(file_paths)):
	plt.plot(avg_avg_data[i],linewidth=2,label=labels[i])
plt.xlabel("Generations (#)")
plt.ylabel("Average of individuals' average win rate")
plt.title("Average of average win rates over generations")

plt.legend(loc="lower center",prop={'size': 8},
# plt.legend(loc="lower right",prop={'size': 6},bbox_to_anchor=(0.1, -0.5),
          fancybox=True, shadow=True, ncol=6,borderaxespad=-8)
plt.savefig(f"testing--.pdf", bbox_inches='tight')
# plt.savefig(f"testing_{seed}.pdf")
# plt.show()

# df = pd.DataFrame(data,columns=["MOVE_FROM_HOME","MOVE_FROM_HOME_AND_KILL","MOVE","MOVE_ONTO_STAR","MOVE_ONTO_STAR_AND_DIE","MOVE_ONTO_STAR_AND_KILL","MOVE_ONTO_GLOBE","MOVE_ONTO_GLOBE_AND_DIE","MOVE_ONTO_ANOTHER_DIE","MOVE_ONTO_ANOTHER_KILL","MOVE_ONTO_VICTORY_ROAD","MOVE_ONTO_GOAL"])
# df.plot(markersize=5,title="Action Values Throughout Generations")
# plt.xlabel("Generations (#)")
# plt.ylabel("Action Value")
# # plt.show()
# plt.savefig("av-plot.pdf")