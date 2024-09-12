import numpy as np


class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.95, epsilon=0.05):
        """
        Initialize the Q-learning agent.

        Parameters:
        - state_size: List representing the size of the state space. For example, if we have 
          3 dimensions for electricity and 2 for gas, it could be [2, 2, 2, 2, 2].
        - action_size: List representing the size of the action space for both electricity and gas appliances.
        - alpha: Learning rate (how fast the agent updates Q-values).
        - gamma: Discount factor (how much future rewards are considered).
        - epsilon: Exploration rate (probability of taking a random action).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table with zeros. The shape of the Q-table is state_size + action_size.
        self.q_table = np.zeros(state_size + action_size)

    def _state_index(self, state):
        """
        Convert the multidimensional state to a tuple for indexing the Q-table.

        Parameters:
        - state: The current state of the environment (list of integers, including electricity and gas states).
        
        Returns:
        - Tuple index corresponding to the given state for Q-table lookup.
        """
        return tuple(state)

    def _action_index(self, action):
        """
        Convert the multidimensional action to a tuple for indexing the Q-table.

        Parameters:
        - action: The action to be taken (list of integers, including actions for electricity and gas).

        Returns:
        - Tuple index corresponding to the given action for Q-table lookup.
        """
        return tuple(action)

    def choose_action(self, state):
        state_index = self._state_index(state)

        if np.random.rand() < self.epsilon:
            # Explore: random action for each appliance
            action = [np.random.randint(self.action_size[i]) for i in range(len(self.action_size))]
            return action

        # Exploit: best action for each appliance
        q_values = self.q_table[state_index]
    
        # Find the index of the best action (the one with the highest Q-value)
        # Since `q_values` may be multidimensional, we flatten it
        best_action_flat_index = np.argmax(q_values.flatten())

        # Convert the flat index back to the multidimensional action index
        best_action = np.unravel_index(best_action_flat_index, self.q_table[state_index].shape)
    
        return list(best_action)


    def learn(self, state, action, reward, next_state):
        """
        Update the Q-table using the Q-learning formula after taking an action.

        Parameters:
        - state: The current state before taking the action.
        - action: The action taken (list of integers, representing both electricity and gas actions).
        - reward: The reward received after taking the action.
        - next_state: The state of the environment after taking the action.
        """
        state_index = self._state_index(state)
        action_index = self._action_index(action)
        next_state_index = self._state_index(next_state)

        # Q-learning update rule: Q(s, a) = Q(s, a) + α * [r + γ * max_a' Q(s', a') − Q(s, a)]
        current_q = self.q_table[state_index][action_index]  # Current Q-value for the taken action
        next_max_q = np.max(self.q_table[next_state_index])  # Max Q-value for the next state

        # Update the Q-value for the action taken
        self.q_table[state_index][action_index] = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)

    def update_epsilon(self, decay_rate):
        """
        Optionally decrease epsilon over time to reduce exploration as the agent learns.

        Parameters:
        - decay_rate: The rate at which epsilon decreases.
        """
        self.epsilon *= decay_rate

    def save_q_table(self, filename):
        """
        Save the Q-table to a file for later use.

        Parameters:
        - filename: The path to the file where the Q-table will be saved.
        """
        np.save(filename, self.q_table)

    def load_q_table(self, filename):
        """
        Load the Q-table from a file.

        Parameters:
        - filename: The path to the file where the Q-table is saved.
        """
        self.q_table = np.load(filename)
