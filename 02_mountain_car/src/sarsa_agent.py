import numpy as np

from src.base_agent import BaseAgent

class SarsaAgent(BaseAgent):

    def __init__(self, env, alpha, gamma):

        self.env = env
        self.q_table = self._init_q_table()

        # hyper-parameters
        self.alpha = alpha
        self.gamma = gamma

    def _init_q_table(self) -> np.array:
        """
        Return numpy array with 3 dimensions.
        The first 2 dimensions are the state components, i.e. position, speed.
        The third dimension is the action.
        """
        # discretize state space from a continuous to discrete
        high = self.env.observation_space.high
        low = self.env.observation_space.low
        n_states = (high - low) * np.array([10, 100])
        n_states = np.round(n_states, 0).astype(int) + 1

        # table with q-values: n_states[0] * n_states[1] * n_actions
        return np.zeros([n_states[0], n_states[1], self.env.action_space.n])

    def _discretize_state(self, state):
        min_states = self.env.observation_space.low
        state_discrete = (state - min_states) * np.array([10, 100])
        return np.round(state_discrete, 0).astype(int)

    def get_action(self, state):
        """"""
        state_discrete = self._discretize_state(state)
        return np.argmax(self.q_table[state_discrete[0], state_discrete[1]])

    def update_parameters(self, state, action, reward, next_state):
        """"""
        s = self._discretize_state(state)
        ns = self._discretize_state(next_state)
        na = self.get_action(next_state)

        delta = self.alpha * (
                reward
                + self.gamma * self.q_table[ns[0], ns[1], na]
                - self.q_table[s[0], s[1], action]
        )
        self.q_table[s[0], s[1], action] += delta