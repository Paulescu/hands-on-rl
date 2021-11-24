import gym

env = gym.make("Taxi-v2").env

env.render()



import random

def train(n_episodes: int):
    """
    Pseudo-code of a Reinforcement Learning agent training loop
    """

    # python object that wraps all environment logic. Typically you will
    # be using OpenAI gym here.
    env = load_env()

    # python object that wraps all agent policy (or value function)
    # parameters, and action generation methods.
    agent = get_rl_agent()

    for episode in range(0, n_episodes):

        # random start of the environmnet
        state = env.reset()

        # epsilon is parameter that controls the exploitation-exploration trade-off.
        # it is good practice to set a decaying value for epsilon
        epsilon = get_epsilon(episode)

        done = False
        while not done:

            if random.uniform(0, 1) < epsilon:
                # Explore action space
                action = env.action_space.sample()
            else:
                # Exploit learned values (or policy)
                action = agent.get_best_action(state)

            # environment transitions to next state and maybe rewards the agent.
            next_state, reward, done, info = env.step(action)

            # adjust agent parameters. Wwe will see how later in the course.
            agent.update_parameters(state, action, reward, next_state)

            state = next_state