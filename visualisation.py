import json
import matplotlib.pyplot as plt
import numpy as np
# Charger les résultats depuis le fichier JSON
with open("q_learning_results.json", "r") as f:
    results = json.load(f)

Q_table = results["Q_table"]  # La Q-table
rewards_per_episode = results["rewards_per_episode"]  # Récompenses par épisode

# Afficher la Q-table
print("Loaded Q-table:")
for row in Q_table:
    print(row)

# Tracer les récompenses par épisode
episodes = range(1, len(rewards_per_episode) + 1)

plt.plot(episodes, rewards_per_episode)
plt.title("Q-Learning Performance Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
# Fonction pour moyenne glissante
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Tracer la moyenne glissante
window_size = 300  # Taille de la fenêtre
smoothed_rewards = moving_average(rewards_per_episode, window_size)

plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed Reward")
plt.title("Q-Learning Performance (Smoothed)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.grid(True)
plt.legend()
plt.show()