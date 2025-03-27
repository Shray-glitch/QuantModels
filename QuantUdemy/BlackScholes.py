from numpy import log, exp, sqrt
from scipy import stats


# S stock price, E strike price, T expiry time, rf risk-free rate, sigma volatility
def call_option_price(S, E, T, rf, sigma):
    # calculate d1 and d2
    d1 = (log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    print("print d1 and d2 params: %s, %s" %(d1, d2))

    # Use N(x) to calculate price of option
    return S*stats.norm.cdf(d1) - E*exp(-rf*T)*stats.norm.cdf(d2)

# S stock price, E strike price, T expiry time, rf risk-free rate, sigma volatility
def put_option_price(S, E, T, rf, sigma):
    # calculate d1 and d2
    d1 = (log(S/E)+(rf+sigma*sigma/2.0)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    print("print d1 and d2 params: %s, %s" %(d1, d2))

    # Use N(x) to calculate price of option
    return -S*stats.norm.cdf(-d1) + E*exp(-rf*T)*stats.norm.cdf(-d2)


if __name__ == '__main__':

    # stock at t=0
    S0 = 100
    # strike price
    E = 100
    # expiry = 1 year = 365 days
    T = 1
    # risk-free rate
    rf = 0.05
    # volatility
    sigma = 0.2

    print("Call option according to Black-Scholes model:", call_option_price(S0, E, T, rf, sigma))
    print("Put option according to Black-Scholes model:", put_option_price(S0, E, T, rf, sigma))