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


data_victor = pd.read_csv("data/elite_performance/cr_0.75_se_tournament/cr_0.75_se_tournament.csv").to_numpy().flatten()
data_peter = np.load('final_run/_evaluation.npy')[:1000,0]
peter_chromosome = np.load("final_run/_best_chromosome.npy")

# print(data_victor)

# data_victor = [x.to_numpy().flatten() for x in data_victor]

data = [data_peter, data_victor]

bartlett_results = stats.bartlett(*data)

print("bartlett test p-value = ",bartlett_results.pvalue)

subset = 1000
statistic, pvalue = stats.ttest_ind(a=data[0], b=data[1], equal_var=True)
# print("t test p-value = ",pvalue)
# print("t test test = ",statistic)

print(f"{pvalue = }")
print(f"{statistic = }")

print("peter mean = ", np.mean(data[0]))
print("victor mean = ",np.mean(data[1]))

plt.hist(data[0],alpha=0.5)
plt.hist(data[1],alpha=0.5)
plt.show()

