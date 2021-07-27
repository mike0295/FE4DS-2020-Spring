from pandas_datareader import data as pdr
import yfinance as yf

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def main():
    data = get_data(save_plt=True)

    data = data.reshape(-1)
    mean = np.mean(data)
    std = np.std(data)

    print("===================================")
    print("Underlying asset: BTC-USD")
    print("\nMean:", mean)
    print("Std:", std)

    plt.hist(data, bins=50, density=True)
    x = np.arange(-0.5, 0.2, 0.0001)
    plt.title("Normal Distribution Fit")
    plt.plot(x, norm.pdf(x, mean, std))
    plt.savefig("normalplot.png")
    
    # 아래 값들은 2021년 5월 24일 기준이다.
    S = 37965
    K = 38000
    r = 0.01
    tau = 18/365
    call, put = black_scholes(S, K, r, std, tau)

    print("\nCall price:", call)
    print("Put price:", put)

    long_straddle(call, put, K)

    print("\nImages saved to current directory.\n")


    print("Probability of profit by price fall:", norm.cdf(1-(K-call)/K, mean, std))
    print("Probability of profit by price rise", 1-norm.cdf((K+put)/K-1, mean, std))
    print("Probability of profit:", norm.cdf(1-(K-call)/K, mean, std)+1-norm.cdf((K+put)/K-1, mean, std))
    print("===================================")

    return


def get_data(save_plt):
    data = []
    start_dates = ["2018-01-17", "2019-07-26"]
    end_dates = ["2019-01-17", "2020-07-26"]

    df1 = pdr.get_data_yahoo("BTC-USD", start=start_dates[0], end=end_dates[0])
    df2 = pdr.get_data_yahoo("BTC-USD", start=start_dates[1], end=end_dates[1])

    df1 = df1.dropna()
    df2 = df2.dropna()

    df1['log_return'] = np.log(df1.Close) - np.log(df1.Close.shift(1))
    df2['log_return'] = np.log(df2.Close) - np.log(df2.Close.shift(1))

    data.append(df1.log_return[1:])
    data.append(df2.log_return[1:])

    data = np.array(data)

    if save_plt is True:
        plt.hist(data[0], bins=50, density=True)
        plt.title("2018-01-17 ~ 2019-01-17")
        plt.savefig("2018-01-17_2019-01-17.png")
        plt.clf()

        plt.hist(data[1], bins=50, density=True)
        plt.title("2019-07-26 ~ 2020-07-26")
        plt.savefig("2019-07-26_2020-07-26.png")
        plt.clf()

        plt.hist(data.reshape(-1), bins=50, density=True)
        plt.title("Combined")
        plt.savefig("combined.png")
        plt.clf()

    return data
    
def black_scholes(S, K, r, std, tau):
    d1 = (np.log(S/K) + (r+0.5*std**2)*tau)/(std*np.sqrt(tau))
    d2 = d1- std*np.sqrt(tau)
    K_t = K * np.exp(-r*tau)

    C_t = S*norm.cdf(d1) - K_t*norm.cdf(d2)
    P_t = K_t*norm.cdf(-d2) - S*norm.cdf(-d1)

    return C_t, P_t

def long_straddle(call, put, strike):
    plt.clf()
    plt.cla
    plt.title("Long Straddle")

    x = np.arange(36000, 40000, 100)
    call_line = np.where(x > strike, x-strike, 0) - call
    put_line = np.where(x < strike, strike-x, 0) - put

    plt.plot(x, call_line, label="Call")
    plt.plot(x, put_line, label="Put")
    plt.plot(x, call_line+put_line, label="Long straddle")

    plt.axhline(y=0, color='black', linestyle='--')

    plt.xlabel("Stock Price")
    plt.ylabel("Profit and Loss")
    plt.legend()
    plt.savefig("long_straddle.png")

if __name__ == '__main__':
    main()
    