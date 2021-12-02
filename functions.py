import numpy as np
import pandas as pd
import pandas_datareader as pdr
import datetime
import requests
import os
from functools import reduce
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# This function queries the TIINGO API returning a dataframe containing the securities' adjusted Closing prices
# Only input needed is the ticker as a string or a list of tickers

def get_stonk_data(tickers):
    all_dataframes=[]
    for i in tickers:
        df = pdr.get_data_tiingo(f'{i}', api_key='fb472f33215bd53309e3193c05ffb5850a7c95f2')
        df[f'{i}']=df.adjClose
        all_dataframes.append(df[f'{i}'])
    merged_securities=reduce(lambda left,right:pd.merge(left,right,on=['date'], how='outer'), all_dataframes)
    return merged_securities
  
  
# Can make a portfolio like such - portfolio=get_stonk_data['GOOG','NVDA','GBX','PLTR']
# Need to set columns for pipeline, as such - portfolio.columns=['GOOG','NVDA','GBX','AAPL']


# This function normalizes the returns
def normalizePortfolio(portfolio):
    normed_port=portfolio/portfolio.iloc[0]
    return normed_port
  
def getArithReturns(portfolio):
    port_daily_ret=portfolio.pct_change(1)
    return port_daily_ret
    
def getLogReturns(portfolio):
    log_ret=np.log(portfolio/portfolio.shift(1))
    return log_ret
  
  
  

def makeAllRandomPortfolioAllocations():
  """
  Makes 15000 randomly generated portfolios with the tickers we feed into get_stonk_data
  Each portfolio has a diferent allocation to each security
  Computes expected return, variance (Volatility), and sharpe ratio for each portfolio
  """
  
  num_ports = 15000
  all_weights = np.zeros((num_ports,len(portfolio.columns)))
  ret_arr = np.zeros(num_ports)
  vol_arr = np.zeros(num_ports)
  sharpe_arr = np.zeros(num_ports)

  for i in range(num_ports):

    # Create Random Weights
    weights = np.array(np.random.random(4))

    # Rebalance Weights
    weights = weights / np.sum(weights)
    
    # Save Weights
    all_weights[i,:] = weights

    # Expected Return
    ret_arr[i] = np.sum((getLogReturns(portfolio).mean() * weights) *252)

    # Expected Variance
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(getLogReturns(portfolio).cov() * 252, weights)))
    
    # Sharpe Ratio
    
    sharpe_arr[i] = ret_arr[i]/vol_arr[i]
    
  print("Max SR")
  print(sharpe_arr.max())
  print("Portfolio with max sharpe ratio")
  print(sharpe_arr.argmax())
  
 

  
  
    


   
