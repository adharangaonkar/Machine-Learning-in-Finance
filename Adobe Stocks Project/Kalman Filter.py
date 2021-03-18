
import numpy as np
import pyflux as pf
import pandas as pd
from scipy.optimize import minimize
from pandas_datareader import DataReader
from datetime import datetime
import matplotlib.pyplot as plt


def Kalman_Filter(param, *args):
    S = Y.shape[0]
    S = S + 1
    "Initialize Params:"
    Z = param[0]
    T = param[1]
    H = param[2]
    Q = param[3]
    "Kalman Filter Starts:"
    u_predict = np.zeros(S)
    u_update = np.zeros(S)
    P_predict = np.zeros(S)
    P_update = np.zeros(S)
    v = np.zeros(S)
    F = np.zeros(S)
    KF_Dens = np.zeros(S)
    for s in range(1, S):
        if s == 1:
            P_update[s] = 1000
            P_predict[s] = T * P_update[1] * np.transpose(T) + Q
        else:
            F[s] = Z * P_predict[s - 1] * np.transpose(Z) + H
            v[s] = Y[s - 1] - Z * u_predict[s - 1]
            u_update[s] = u_predict[s - 1] + P_predict[s - 1] * np.transpose(Z) * (1 / F[s]) * v[s]
            u_predict[s] = T * u_update[s];
            P_update[s] = P_predict[s - 1] - P_predict[s - 1] * np.transpose(Z) * (1 / F[s]) * Z * P_predict[s - 1];
            P_predict[s] = T * P_update[s] * np.transpose(T) + Q
            KF_Dens[s] = (1 / 2) * np.log(2 * np.pi) + (1 / 2) * np.log(abs(F[s])) + (1 / 2) * np.transpose(v[s]) * (
                        1 / F[s]) * v[s]
            Likelihood = sum(KF_Dens[1:-1])

            return Likelihood


def Kalman_Smoother(params, Y, *args):
    S = Y.shape[0]
    S = S + 1
    "Initialize Params:"
    Z = params[0]
    T = params[1]
    H = params[2]
    Q = params[3]
    "Kalman Filter Starts:"
    u_predict = np.zeros(S)
    u_update = np.zeros(S)
    P_predict = np.zeros(S)
    P_update = np.zeros(S)
    v = np.zeros(S)
    F = np.zeros(S)
    for s in range(1, S):
        if s == 1:
            P_update[s] = 100
            P_predict[s] = T * P_update[1] * np.transpose(T) + Q
        else:
            F[s] = Z * P_predict[s - 1] * np.transpose(Z) + H
            v[s] = Y[s - 1] - Z * u_predict[s - 1]
            u_update[s] = u_predict[s - 1] + P_predict[s - 1] * np.transpose(Z) * (1 / F[s]) * v[s]
            u_predict[s] = T * u_update[s];
            P_update[s] = P_predict[s - 1] - P_predict[s - 1] * np.transpose(Z) * (1 / F[s]) * Z * P_predict[s - 1];
            P_predict[s] = T * P_update[s] * np.transpose(T) + Q

            u_smooth = np.zeros(S)
            P_smooth = np.zeros(S)
            u_smooth[S - 1] = u_update[S - 1]
            P_smooth[S - 1] = P_update[S - 1]
    for t in range(S - 1, 0, -1):
        u_smooth[t - 1] = u_update[t] + P_update[t] * np.transpose(T) / P_predict[t] * (u_smooth[t] - T * u_update[t])
        P_smooth[t - 1] = P_update[t] + P_update[t] * np.transpose(T) / P_predict[t] * (P_smooth[t] - P_predict[t]) / \
                          P_predict[t] * T * P_update[t]
    u_smooth = u_smooth[0:-1]
    return u_smooth


adobe = DataReader('ADBE', 'yahoo', datetime(2018, 1, 1), datetime(2020, 1, 1))
Y = adobe['Adj Close'].values
T = Y.size;
param0 = np.array([0.9, 0.9, np.std(Y), np.std(Y)])
param_star = minimize(Kalman_Filter, param0, method='BFGS', options={'xtol': 1e-8, 'disp': True})
u = Kalman_Smoother(param_star.x, Y)
timevec = np.linspace(1, T, T)
print(np.std(Y))
plt.plot(timevec, u, 'r', timevec, (Y)*0.32, 'b:')
plt.show()