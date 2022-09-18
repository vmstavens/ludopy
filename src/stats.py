#!/bin/python3
from turtle import shape
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sympy import false, latex
import glob
import os
from scipy.stats import kstest, norm
from sklearn.metrics import dcg_score
from sklearn.metrics import ndcg_score


# configs
PLOT     = False
SHAPIRO  = False
BARTLETT = True
ANOVA_2  = True 
BINS     = 100
QQ       = False
DCG      = False

# folder
folder = "data/elite_performance/"
# folder = "data/indi_performance/"

files = []

for path in os.listdir(folder):
	# print(path)
	full_path = os.path.join(folder, path)
	file = full_path + "/" + path + ".csv"
	files.append(file)

cr_75_se_tour = pd.read_csv(files[0],header=None)
cr_25_se_roul = pd.read_csv(files[1],header=None)
cr_25_se_tour = pd.read_csv(files[2],header=None)
cr_50_se_roul = pd.read_csv(files[3],header=None)
cr_75_se_roul = pd.read_csv(files[4],header=None)
cr_50_se_tour = pd.read_csv(files[5],header=None)

data = [cr_25_se_tour,
		cr_50_se_tour,
		cr_75_se_tour,
		cr_25_se_roul,
		cr_50_se_roul,
		cr_75_se_roul]

# subset = 10

# anova_data = [
# 				cr_25_se_tour[:subset],
# 				cr_50_se_tour[:subset],
# 				cr_75_se_tour[:subset],
# 				cr_25_se_roul[:subset],
# 				cr_50_se_roul[:subset],
# 				cr_75_se_roul[:subset]
# 			]

# data = anova_data


print("cr_25_tour : mean = {:.9f} std = {:.9f}".format( cr_25_se_tour.to_numpy().flatten().mean(), cr_25_se_tour.to_numpy().flatten().std() ))
print("cr_50_tour : mean = {:.9f} std = {:.9f}".format( cr_50_se_tour.to_numpy().flatten().mean(), cr_50_se_tour.to_numpy().flatten().std() ))
print("cr_75_tour : mean = {:.9f} std = {:.9f}".format( cr_75_se_tour.to_numpy().flatten().mean(), cr_75_se_tour.to_numpy().flatten().std() ))
print("cr_25_roul : mean = {:.9f} std = {:.9f}".format( cr_25_se_roul.to_numpy().flatten().mean(), cr_25_se_roul.to_numpy().flatten().std() ))
print("cr_50_roul : mean = {:.9f} std = {:.9f}".format( cr_50_se_roul.to_numpy().flatten().mean(), cr_50_se_roul.to_numpy().flatten().std() ))
print("cr_75_roul : mean = {:.9f} std = {:.9f}".format( cr_75_se_roul.to_numpy().flatten().mean(), cr_75_se_roul.to_numpy().flatten().std() ))

columns = ["crossover_rate","selection_method","win_rate"]
crs     = ["0.25","0.50","0.75"]
meth    = ["roulette","tournament"]
df      = pd.DataFrame(columns=columns)

for d in data[0].values:
	temp_df = pd.DataFrame({columns[0]:crs[0],columns[1]:meth[0],columns[2]:d})
	df = pd.concat([df, temp_df])
 
for d in data[1].values:
	temp_df = pd.DataFrame({columns[0]:crs[1],columns[1]:meth[0],columns[2]:d})
	df = pd.concat([df, temp_df])
 
for d in data[2].values:
	temp_df = pd.DataFrame({columns[0]:crs[2],columns[1]:meth[0],columns[2]:d})
	df = pd.concat([df, temp_df])
 
for d in data[3].values:
	temp_df = pd.DataFrame({columns[0]:crs[0],columns[1]:meth[1],columns[2]:d})
	df = pd.concat([df, temp_df])
 
for d in data[4].values:
	temp_df = pd.DataFrame({columns[0]:crs[1],columns[1]:meth[1],columns[2]:d})
	df = pd.concat([df, temp_df])
 
for d in data[5].values:
	temp_df = pd.DataFrame({columns[0]:crs[2],columns[1]:meth[1],columns[2]:d})
	df = pd.concat([df, temp_df])

# fix data type for values
df[columns[2]] = pd.to_numeric(df[columns[2]]) 

labels = ["crossover rate 0.25, roulette selection",
			"crossover rate 0.50, roulette selection",
			"crossover rate 0.75, roulette selection",
			"crossover rate 0.25, tournament selection",
			"crossover rate 0.50, tournament selection",
			"crossover rate 0.75, tournament selection"]

shapiro_results  = []
bartlett_results = []

new_data = [x.to_numpy().flatten() for x in data]

if PLOT:

	plt.hist(data[0], BINS, alpha=0.5, label=labels[0])
	plt.hist(data[1], BINS, alpha=0.5, label=labels[1])
	plt.hist(data[2], BINS, alpha=0.5, label=labels[2])
	plt.hist(data[3], BINS, alpha=0.5, label=labels[3])
	plt.hist(data[4], BINS, alpha=0.5, label=labels[4])
	plt.hist(data[5], BINS, alpha=0.5, label=labels[5])
	plt.legend(loc='upper right')
	plt.show()

if SHAPIRO:
	for sol in data:
		shapiro_results.append(stats.shapiro(sol))
	print("p-value = \n", [x.pvalue for x in shapiro_results] )

if BARTLETT:
	bartlett_results = stats.bartlett(*new_data)
	print("bartlett test p-value = ",bartlett_results.pvalue)
	if bartlett_results.pvalue > 0.05:
		print("EQUAL VARIANCE")
	else:
		print("NOT EQUAL VARIANCE")

if ANOVA_2:
	formula = f"{columns[2]} ~ C({columns[0]}) + C({columns[1]}) + C({columns[0]}):C({columns[1]})"
	model = ols(formula=formula,data=df).fit()
	# a = sm.stats.anova_lm(model, typ=2)
	print(sm.stats.anova_lm(model, typ=2).to_latex())


if DCG:
	# Corpus# True relevance score - scale from 0-10
	true_relevance = {'d1': 10, 'd2': 9, 'd3':7, 'd4':6, 'd5':4}# Predicted relevence score
	predicted_relevance = {'d1': 8, 'd2': 9, 'd3':6, 'd4':6, 'd5':5}# relevance list processed as array
	true_rel = np.asarray([list(true_relevance.values())])
	predicted_rel = np.asarray([list(predicted_relevance.values())])



subset = 1000
