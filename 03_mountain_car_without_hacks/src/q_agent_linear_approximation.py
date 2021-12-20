import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

from pdb import set_trace as stop

class QAgentLinearApproximation:

    def __init__(self, env, learning_rate: float, discount_factor: float):

        self.env = env

        # hyper-parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.preprocessor = self._init_preprocessor()
        self.models = self._init_linear_models()

    def _init_preprocessor(self):

        # normalize position and velocity to have zero mean and unit variance
        scaler = StandardScaler()

        # feature engineering
        featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
        ])

        preprocessor = Pipeline([
            ('scaler', scaler),
            ('featurizer', featurizer)
        ])

        observation_examples = [self.env.observation_space.sample() for x in range(100)]

        return preprocessor.fit(observation_examples)

    def _init_linear_models(self):

        models = []
        for _ in range(0, self.env.action_space.n):
            model = SGDRegressor(learning_rate='constant', eta0=self.learning_rate)

            # 0 is our initial estimate of the q-values. It is a very optimistic
            # estimate, as the default per step reward is -1
            model.partial_fit([self._preprocess_state(self.env.reset())], [0])

            models.append(model)

        return models

    def _preprocess_state(self, state):
        return self.preprocessor.transform([state])[0]

    def _get_q_values(self, state) -> np.array:

        q_values = np.array([self.models[i].predict([self._preprocess_state(state)])
                             for i in range(self.env.action_space.n)])
        return q_values.reshape(-1)

    def update_parameters(self, state, action, reward, next_state):

        # q-learning target
        q_values_next_state = self._get_q_values(next_state)
        target = reward + self.discount_factor * np.max(q_values_next_state)

        # one step of gradient descent towards the target
        self.models[action].partial_fit([self._preprocess_state(state)], [target])

    def get_action(self, state):

        q_values = self._get_q_values(state)
        return np.argmax(q_values)


if __name__ == '__main__':

    import gym
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000

    alpha = 0.01 # learning rate
    gamma = 0.9 # discount factor
    agent = QAgentLinearApproximation(env, alpha, gamma)

    from src.loops import train
    for i in range(0, 100):
        rewards, max_positions = train(agent, env, n_episodes=10, epsilon=0.10)
        print(rewards)
        print(max_positions)


