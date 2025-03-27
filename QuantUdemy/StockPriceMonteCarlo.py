import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


NUM_OF_SIMULATIONS = 100
# N == number of days essentially
def stock_monty_carlo(S0, mu, sigma, N=1000):

    results = []

    # number of simulations
    for _ in range(NUM_OF_SIMULATIONS):
        prices = [S0]
        for _ in range(N):
            # simulate change day by day, so t=1
            stock_price = prices[-1] * np.exp((mu - 0.5 * sigma ** 2) +
                                              sigma * np.random.normal())

            prices.append(stock_price)

        results.append(prices)

    simulation_data = pd.DataFrame(results)
    # given columns will contain time series for a given simulation
    simulation_data = simulation_data.T # T = transpose
    # print(simulation_data)

    # calculate mean of all simulations
    simulation_data['mean'] = simulation_data.mean(axis=1)
    # print(simulation_data)

    print("Prediction for future stock price: $ %.2f" % simulation_data['mean'].tail(1).iloc[0] )

    # will plot the mean of all simulations
    plt.plot(simulation_data['mean'])
    plt.show()

if __name__ == '__main__':

    stock_monty_carlo(50, 0.0002, 0.01)