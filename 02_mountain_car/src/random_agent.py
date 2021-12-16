from src.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    This taxi driver selects actions randomly.
    You better not get into this taxi!
    """
    def __init__(self, env):
        self.env = env

    def get_action(self, state) -> int:
        """
        No input arguments to this function.
        The agent does not consider the state of the environment when deciding
        what to do next.
        """
        return self.env.action_space.sample()

    def update_parameters(self, state, action, reward, next_state):
        pass

