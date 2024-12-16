import numpy as np
import json
import matplotlib.pyplot as plt
# Define the environment
n_states = 16  # Number of states in the grid world
n_actions = 4  # Actions: up, down, left, right
goal_state = 15  # Goal state

# Initialize Q-table with small random values
Q_table = np.random.rand(n_states, n_actions) * 0.01

# Parameters
learning_rate = 0.3   #(𝛼)
discount_factor = 0.99   #(𝛾)
epochs = 1000
epsilon = 1.0  # Start with 100% exploration
epsilon_decay = 0.99  # Decay rate for epsilon
epsilon_min = 0.01  # Minimum exploration probability

# Track rewards per episode
rewards_per_episode = []

# Helper function: Transition to the next state based on action
def get_next_state(state, action):
    grid_size = int(np.sqrt(n_states))  # Assuming a 4x4 grid
    row, col = divmod(state, grid_size)

    if action == 0:  # UP
        row = max(0, row - 1)
    elif action == 1:  # DOWN
        row = min(grid_size - 1, row + 1)
    elif action == 2:  # LEFT
        col = max(0, col - 1)
    elif action == 3:  # RIGHT
        col = min(grid_size - 1, col + 1)
    
    return row * grid_size + col  # Return the new state index

max_steps = 10  # Plus de pas autorisés

# Q-learning algorithm
for epoch in range(epochs):
    current_state = np.random.randint(0, n_states)  # Start from a random state
    total_reward = 0  # Track total reward
    epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon
    steps =0
    while current_state != goal_state and steps < max_steps:
        steps =+ 1
        # Choose action using epsilon-greedy strategy
        if np.random.rand() < epsilon:
            action = np.random.randint(0, n_actions)  # Explore
        else:
            action = np.argmax(Q_table[current_state])  # Exploit

        # Get next state and reward
        next_state = get_next_state(current_state, action)
        reward = -0.1  # Penalty for each step
        if next_state == goal_state:
            reward = 1.0  # Reward for reaching the goal

        total_reward += reward  # Accumulate reward

        # Update Q-value using Q-learning formula
        Q_table[current_state, action] += learning_rate * (
            reward + discount_factor * np.max(Q_table[next_state]) - Q_table[current_state, action]
        )

        current_state = next_state  # Move to the next state

    # Force the Q-values of the goal state to zero
    Q_table[goal_state] = [0, 0, 0, 0]

    # Track total reward for this episode
    rewards_per_episode.append(total_reward)

# Display the learned Q-table
print("Learned Q-table:\n['UP', 'DOWN', 'LEFT', 'RIGHT']")
print(Q_table)

# Save results to JSON
results = {
    "Q_table": Q_table.tolist(),
    "rewards_per_episode": rewards_per_episode
}

with open("q_learning_results.json", "w") as f:
    json.dump(results, f)

print("Results saved to q_learning_results.json")






# Grille et actions


grid_size = 4
actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
optimal_policy = np.argmax(Q_table, axis=1)

# Créer une figure
fig, ax = plt.subplots(figsize=(5, 5))

# Dessiner la grille
for i in range(grid_size):
    for j in range(grid_size):
        state = i * grid_size + j

        # Vérifier si l'état est terminal (toutes les Q-valeurs = 0)
        if np.all(Q_table[state] == 0):
            ax.scatter(j, grid_size - i - 1, color='green', s=100)  # Afficher un point
        else:
            action = actions[optimal_policy[state]]
            # Ajouter des flèches pour représenter les actions optimales
            if action == 'UP':
                ax.arrow(j, grid_size - i - 1, 0, 0.3, head_width=0.1)
            elif action == 'DOWN':
                ax.arrow(j, grid_size - i - 1, 0, -0.3, head_width=0.1)
            elif action == 'LEFT':
                ax.arrow(j, grid_size - i - 1, -0.3, 0, head_width=0.1)
            elif action == 'RIGHT':
                ax.arrow(j, grid_size - i - 1, 0.3, 0, head_width=0.1)

# Afficher les actions optimales dans la console
for i in range(grid_size):
    row = optimal_policy[i * grid_size: (i + 1) * grid_size]
    row_actions = []
    for idx, state in enumerate(row):
        if np.all(Q_table[i * grid_size + idx] == 0):
            row_actions.append('END')
        else:
            row_actions.append(actions[state])
    print(" ".join(row_actions))


for state, action_idx in enumerate(optimal_policy):
    # Vérifier si l'état est terminal (toutes les valeurs Q sont nulles)
    if np.all(Q_table[state] == 0):
        print(f"State {state}: Best Action -> END")
    else:
        print(f"State {state}: Best Action -> {actions[action_idx]}")
# Configurer l'affichage
ax.set_xlim(-0.5, grid_size - 0.5)
ax.set_ylim(-0.5, grid_size - 0.5)
ax.set_xticks(np.arange(grid_size))
ax.set_yticks(np.arange(grid_size))
ax.grid(True)
plt.title("Politique Optimale (Q-Learning)")
plt.show()