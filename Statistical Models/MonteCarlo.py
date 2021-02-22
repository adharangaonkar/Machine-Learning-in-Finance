#MONTE CARLO

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas_datareader import DataReader
from datetime import datetime


AAPL = DataReader('AAPL',  'yahoo', datetime(2020,1,1), datetime(2020,12,31))
X = AAPL['Adj Close'].values

sns.kdeplot(data=X)

T = X.shape[0]
M = 1000
mu = np.mean(X)
se = np.std(X)
mu_mc = np.zeros(M)
se_mc = np.zeros(M)
t_stat_mc = np.zeros(M)
for i in range(0, M):
    y_mc = mu + se*np.random.normal(0,4,T)
    mu_mc[i] = np.mean(y_mc)
    se_mc[i] = np.std(y_mc)/np.sqrt(T)
mu_mc = np.sort(mu_mc)
se_mc = np.sort(se_mc)/np.sqrt(T)

print("confidence interval of mu_mc:", mu_mc[25], mu_mc[975])

plt.show()