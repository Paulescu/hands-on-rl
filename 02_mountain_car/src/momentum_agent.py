from src.base_agent import BaseAgent

class MomentumAgent(BaseAgent):

    def __init__(self, env):
        self.env = env

        self.valley_position = -0.5

    def get_action(self, state) -> int:
        """
        No input arguments to this function.
        The agent does not consider the state of the environment when deciding
        what to do next.
        """
        velocity = state[1]

        if velocity > 0:
            # accelerate to the right
            action = 2
        else:
            # accelerate to the left
            action = 0

        return action

    def update_parameters(self, state, action, reward, next_state):
        pass

