#Marketing making; mean reversion; Ornstein-Uhlenbeck Process (OU); 
#This algorithm was written based Market Making and Mean Reversion by Tanmoy Chakraborty and Michael Kearns

import numpy as np
import matplotlib.pyplot as plt
# Parameters for the Ornstein-Uhlenbeck process
gamma = 0.1 # Mean reversion rate
sigma = 1.0 # Volatility
mu = 100.0 # Long-term mean price
T = 1000 #Time horizon
dt = 1 #Time step
Q0 = mu #initial price

# Function to simulate the Ornstein-Uhlenbeck process
def simulate_ou_process(gamma, sigma, mu, T, dt, Q0):
  num_steps = int(T/dt)
  Q = np.zeros(num_steps)
  Q[0] = Q0
  for t in range(1, num_steps):
    dW = np.random.normal(0, 
                          np.sqrt(dt)) # Wiener process increment
    dQ = -gamma * (Q[t-1]-mu)* dt + sigma * dW
    Q[t] = Q[t-1] + dQ
  return Q

# Market making algorithm
def market_making_algorithm(Q,
                            spread=1):
    inventory = 0
    cash = 0
    profits = []
    for t in range(len(Q)):
      # Place buy and sell orders around the current price
        buy_price = Q[t] - spread
        sell_price = Q[t] + spread
      
#Simulate market orders hitting our quotes
#Assume a random market order hits our quotes with some probability
        if np.random.rand() < 0.5: #Buy order hits our sell quote
          inventory-= 1
          cash += sell_price
        if np.random.rand() < 0.5: # sell order hits our buy quote
          inventory += 1
          cash -= buy_price
#Record profit at each step
        profit = cash + inventory * Q[t]profits.append(profit)
    return profits

#Simulate the OU process
Q = simulate_ou_process(gamma, sigma, mu, T, dt, Q0)

# Run the market making algorithm
profits = market_making_algorithm(Q)

#Plot the results
plt.figure(figsize=(10, 6))
plt.plot(Q, label='Price (OU Process)')
plt.plot(profits, label='Profit')
plt.xlabel('Time')
plt.ylabel ('Value')
plt.title('Market Making Algorithm Performance')
plt.legend()
plt.show()
          
    
                              
    


