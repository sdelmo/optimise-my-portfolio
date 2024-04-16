"""
Author: Sebastian Delgado
Description: Using Py to optimize a portfolio using Modern Portfolio Theory
Background: Modern Portfolio Theory is a hypothesis put forth by Harry Markowitz
based on the idea that risk-averse investors can construct portfolios to optimize
or maximize expected return based on a given level of market risk, emphasizing
that risk is an inherent part of higher rewards.


"""

import numpy as np
import pandas as pd
import quandl
import matplotlib.pyplot as plt
from scipy.optimize import minimize



def getData():
    path = "HKEX/"
    start = "2020-01-01"
    end = "2021-01-01"
    samplefreq = "daily"

    # for ticker in tickers:
    #     ticker=quandl.get(path+ticker,start_date,end_date,collapse=samplefreq)

    tencentRaw = quandl.get("HKEX/00419", start_date=start,
                            end_date=end, collapse=samplefreq)
    meituanRaw = quandl.get("HKEX/03690", start_date=start,
                            end_date=end, collapse=samplefreq)
    bydRaw = quandl.get("HKEX/00285", start_date=start,
                        end_date=end, collapse=samplefreq)
    xiaomiRaw = quandl.get("HKEX/01810", start_date=start,
                           end_date=end, collapse=samplefreq)
    smicRaw = quandl.get("HKEX/11012", start_date=start,
                         end_date=end, collapse=samplefreq)

    tencent = tencentRaw["Previous Close"]
    meituan = meituanRaw["Previous Close"]
    byd = bydRaw["Previous Close"]
    xiaomi = xiaomiRaw["Previous Close"]
    smic = smicRaw["Previous Close"]

    stocks = pd.concat([smic, byd, xiaomi, tencent, meituan], axis=1)
    stocks.colums = ['smic', 'byd', 'xiaomi', 'tencent', 'meituan']
    return stocks


def getMeanDailyRet(stocks):
    stocks.pct_change(1).mean()


def getDailyRet(stocks):
    stocks.pct_change(1)


def getReturnCorr(stocks):
    stocks.pct_change(1).corr()


def normalizeStonks(stocks):
    stock_normed = stocks/stocks.iloc[0]
    stock_normed.plot()


def getLogReturns(stocks):
    log_ret = np.log(stocks/stocks.shift(1))


def singleRun(stocks):

    # Stock Columns
    print('Stocks')
    print(stocks.columns)
    print('\n')

    # Create Random Weights
    print('Creating Random Weights')
    weights = np.array(np.random.random(4))
    print(weights)
    print('\n')

    # Rebalance Weights
    print('Rebalance to sum to 1.0')
    weights = weights / np.sum(weights)
    print(weights)
    print('\n')

    # Expected Return
    print('Expected Portfolio Return')
    exp_ret = np.sum(log_ret.mean() * weights) * 252
    print(exp_ret)
    print('\n')

    # Expected Variance
    print('Expected Volatility')
    exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    print(exp_vol)
    print('\n')

    # Sharpe Ratio
    SR = exp_ret/exp_vol
    print('Sharpe Ratio')
    print(SR)


def monteCarlo():
    num_ports = 15000

    all_weights = np.zeros((num_ports, len(stocks.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports):

        # Create Random Weights
        weights = np.array(np.random.random(4))

        # Rebalance Weights
        weights = weights / np.sum(weights)

        # Save Weights
        all_weights[ind, :] = weights

        # Expected Return
        ret_arr[ind] = np.sum((log_ret.mean() * weights) * 252)

        # Expected Variance
        vol_arr[ind] = np.sqrt(
            np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

        # Sharpe Ratio
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

        return vol_arr, ret_arr, sharpe_arr


def plotMC(vol_arr, ret_arr, sharpe_arr):

    max_sr_ret = ret_arr[1419]
    max_sr_vol = vol_arr[1419]

    plt.figure(figsize=(12, 8))
    plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Volatility')
    plt.ylabel('Return')

    # Add red dot for max SR
    plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')
    plt.show()


def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])


def neg_sharpe(weights):
    return get_ret_vol_sr(weights)[2] * -1

# Contraints


def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1


# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type': 'eq', 'fun': check_sum})

# 0-1 bounds for each weight
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))

# Initial Guess (equal distribution)
init_guess = [0.25, 0.25, 0.25, 0.25]

# Sequential Least SQuares Programming (SLSQP).
opt_results = minimize(neg_sharpe, init_guess,
                       method='SLSQP', bounds=bounds, constraints=cons)
