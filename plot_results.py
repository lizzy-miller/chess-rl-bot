# %% 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results = pd.read_csv('mf1_results/results.csv')

# Compute Cumulative Results

games = np.arange(1, len(results) + 1)
win_rate = results["Wins"].cumsum() / games
loss_rate = results["Losses"].cumsum() / games
draw_rate = results["Draws"].cumsum() / games

# Plot
plt.figure(figsize=(10, 6))
plt.plot(games, win_rate, label='Win Rate', color='green')
plt.plot(games, loss_rate, label='Loss Rate', color='red')
plt.plot(games, draw_rate, label='Draw Rate', color='blue')

plt.title('Training Progress Over Time')

plt.xlabel('Games Played')
plt.ylabel('Rate')
plt.ylim(0, 1)
plt.legend()
#plt.savefig('plots/training_progress.png')
plt.show()

# %%
