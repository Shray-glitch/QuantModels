import numpy as np
import matplotlib.pyplot as plt

# Parameters
strike_price = 100
premium = 5
spot_prices = np.linspace(50, 150, 100)

# Payoff and profit
payoff = np.maximum(strike_price - spot_prices, 0)
profit = payoff - premium

# Plot
plt.figure(figsize=(10, 6))
plt.plot(spot_prices, profit, label='Put Option Profit', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.axvline(strike_price, color='gray', linestyle='--', label='Strike Price')
plt.title('Payoff Diagram - Long Put Option')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit')
plt.legend()
plt.grid(True)
plt.show()