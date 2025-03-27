import numpy as np
import yfinance as yf
from scipy.stats import norm
import pandas as pd
import datetime


def download_data(stock, start_date, end_date):

    data = {}
    ticker = yf.download(stock, start_date, end_date, auto_adjust=True)
    data = ticker['Close']
    return pd.DataFrame(data)

# c is confidence level, z is calculated from distribution function, v = alpha
# this is how we calculate VaR tomorrow(n=1)
def calculate_var(position, c, mu, sigma):
    var = position * (mu - sigma * norm.ppf(1-c))
    return var

# for any day in future
def calculate_var_in_n_days(position, c, mu, sigma, n):
    var = position * (mu * n  - sigma * np.sqrt(n) * norm.ppf(1-c))
    return var


if __name__ == '__main__':

    start = datetime.datetime(2017,1, 1)
    end = datetime.datetime(2025, 3, 26)

    stock_data = download_data('ORCL', start, end)

    stock_data['returns'] = np.log(stock_data['ORCL'] / stock_data['ORCL'].shift(1))
    stock_data = stock_data[1:]
    # print(stock_data)

    # Investment
    S = 1e6
    # Confidence level - 95%
    c = 0.95

    # we assume returns are normally dist
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])

    print("Value at Risk is: %0.2f" % calculate_var(S, c, mu, sigma))
    print("Value at Risk after 10 days: $%0.2f" % calculate_var_in_n_days(S, c, mu, sigma, 10))