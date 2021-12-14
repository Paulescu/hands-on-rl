import pickle
from pathlib import Path
from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def update_parameters(self, state, action, reward, next_state):
        pass

    def save_to_disk(self, path: Path):
        """
        Saves python object to disk using a binary format
        """
        with open(path, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_disk(cls, path: Path):
        """
        Loads binary format into Python object.
        """
        with open(path, "rb") as f:
            dump = pickle.load(f)

        return dump