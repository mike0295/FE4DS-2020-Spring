from pandas_datareader import data as pdr
import yfinance as yf
from tqdm import tqdm

import gurobipy as gp
from gurobipy import GRB

from math import sqrt
import pandas as pd
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

TICKER_LIST = ['DAL', 'AAL', 'UAL', 'NFLX', 'DIS', 'ROKU', 'NIO', 'PLUG', 'ARKF', 'ARKW', 'IVV']


def main():
    parser = argparse.ArgumentParser(description="Portfolio Manager")
    parser.add_argument('--start_date', default="2020-01-01")
    parser.add_argument('--end_date', default="2021-01-01") 
    parser.add_argument('--tickers', type=str, nargs='+', default=TICKER_LIST)
    parser.add_argument('--target_return', default=0.0025)

    args = parser.parse_args()
    print("Asset list: ", args.tickers)
    print("Start date: ", args.start_date)
    print("End date: ", args.end_date)
    print("Target return: ", args.target_return)

    data = collect_data(args.tickers, args.start_date, args.end_date)
    frontier = portfolio(data, args.target_return)

    return


def collect_data(asset_list, start_date, end_date):
    print("\nCollecting data begin")
    data = pd.DataFrame()

    try:
        for ticker in tqdm(asset_list):
            _df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
            df = _df.dropna()
            assert df.size == _df.size

            df['log_return'] = np.log(df.Close) - np.log(df.Close.shift(1))
            data[ticker] = df.log_return[1:]

    except:
        print("Invalid ticker or time-range")
        exit()

    print("\nCollecting data complete")

    return data


def portfolio(data, target_return):
    stocks = data.columns
    stock_vol = data.std()
    stock_ret = data.mean()
    sigma = data.cov()
    print(stocks)
    # Create model
    m = gp.Model('portfolio')

    # Create \theta
    vars = pd.Series(m.addVars(stocks), index=stocks)

    # Portfolio risk. Target to be minimized
    p_risk = sigma.dot(vars).dot(vars)

    # Calculates minimum-risk return and volatility
    p_ret, min_ret, min_vol = minimum_risk_portfolio(m, vars, stocks, stock_ret, stock_vol, p_risk)
    
    frontier = efficient_frontier(m, vars, stocks, stock_ret, stock_vol, p_ret, p_risk, min_ret, min_vol, target_return)

    return frontier

def minimum_risk_portfolio(m, vars, stocks, stock_return, stock_volatility, portfolio_risk):
    # Constraint: sum of weights must equal to 1
    m.addConstr(vars.sum() == 1, 'budget')

    m.setObjective(portfolio_risk, GRB.MINIMIZE)

    # Optimize model to find  minimum risk portfolio
    m.setParam('OutputFlag', 0)
    m.optimize()

    # Setting an equation for portfolio return
    portfolio_return = stock_return.dot(vars)

    # Display minimum risk portfolio
    print('\n\nMinimum Risk Portfolio:\n')
    for s, v in zip(stocks, vars):
        if v.x > 0:
            print(s, v.x)
    minrisk_volatility = sqrt(portfolio_risk.getValue())
    minrisk_return = portfolio_return.getValue()
    print('\nVolatility      =', minrisk_volatility)
    print('Expected Return =', minrisk_return)

    return portfolio_return, minrisk_return, minrisk_volatility


def efficient_frontier(m, vars, stocks, stock_return, stock_volatility, portfolio_return, portfolio_risk, minrisk_return, minrisk_volatility, target_return):
    # Add constraint such that return = specific return
    target = m.addConstr(portfolio_return == minrisk_return, 'target')

    # Calculate efficient frontier
    frontier = pd.Series(dtype=np.float64)
    for r in np.linspace(stock_return.min(), stock_return.max(), 100):
        target.rhs = r
        m.optimize()
        frontier.loc[sqrt(portfolio_risk.getValue())] = r

    # Plot volatility versus expected return for individual assets
    ax = plt.gca()
    ax.scatter(x=stock_volatility, y=stock_return, color='Blue', label='Individual Assets')
    for i, stock in enumerate(stocks):
        ax.annotate(stock, (stock_volatility[i], stock_return[i]))

    # Plot volatility versus expected return for minimum risk portfolio
    ax.scatter(x=minrisk_volatility, y=minrisk_return, color='DarkGreen')
    ax.annotate('Minimum\nRisk\nPortfolio', (minrisk_volatility, minrisk_return), horizontalalignment='right')

    # Plot efficient frontier
    frontier.plot(color='DarkGreen', label='Efficient Frontier', ax=ax)

    # Format and display the final plot
    ax.axis([0.005, 0.075, -0.005, 0.015])
    ax.set_xlabel('Volatility (standard deviation)')
    ax.set_ylabel('Expected Return')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig('portfolio.png')
    print("Plotted efficient frontier to 'portfolio.png'")

    target.rhs = target_return
    m.optimize()
    print('\nTarget Return Portfolio:\n')
    for s, v in zip(stocks, vars):
        if v.x > 0:
            print(s, v.x)
    target_volatility = sqrt(portfolio_risk.getValue())
    target_return = portfolio_return.getValue()
    print('\nVolatility      =', target_volatility)
    print('Expected Return =', target_return)

    return frontier
    


if __name__ == "__main__":
    main()