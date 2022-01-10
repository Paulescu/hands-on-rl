import os
from typing import Callable, Dict, Tuple, List
import pathlib
from pathlib import Path
import json
from pdb import set_trace as stop

import numpy as np
import gym
import yaml
import torch.nn as nn


def snake_to_camel(word):
    import re
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


def get_agent_id(env_name: str) -> str:
    """"""
    from src.config import SAVED_AGENTS_DIR

    dir = Path(SAVED_AGENTS_DIR) / env_name
    if not dir.exists():
        os.makedirs(dir)

    try:
        agent_id = max([int(id) for id in os.listdir(dir)]) + 1
    except ValueError:
        agent_id = 0

    return str(agent_id)

def get_input_output_dims(env_name: str) -> Tuple[int]:
    """"""
    if 'MountainCar' in env_name:
        input_dim = 2
        output_dim = 3
    elif 'CartPole' in env_name:
        input_dim = 4
        output_dim = 2
    else:
        raise Exception('Invalid environment')

    return input_dim, output_dim


def get_epsilon_decay_fn(
    eps_start: float,
    eps_end: float,
    total_episodes: int
) -> Callable:
    """
    Returns function epsilon_fn, which depends on
    a single input, step, which is the current episode
    """
    def epsilon_fn(episode: int) -> float:
        r = max((total_episodes - episode) / total_episodes, 0)
        return (eps_start - eps_end)*r + eps_end

    return epsilon_fn


def get_epsilon_exponential_decay_fn(
    eps_max: float,
    eps_min: float,
    decay: float,
) -> Callable:
    """
    Returns function epsilon_fn, which depends on
    a single input, step, which is the current episode
    """
    def epsilon_fn(episode: int) -> float:
        return max(eps_min, eps_max * (decay ** episode))
    return epsilon_fn


def load_default_hyperparameters(env_name: str) -> Dict:
    """"""
    current_dir = pathlib.Path(__file__).parent

    with open(current_dir / 'hyperparameters.yaml', 'r') as stream:
        default_hp = yaml.safe_load(stream)

    return default_hp[env_name]

def get_success_rate_from_n_steps(env: gym.Env, steps: List[int]):

    import numpy as np
    if 'MountainCar' in str(env):
        success_rate = np.mean((np.array(steps) < env._max_episode_steps) * 1.0)
    elif 'CartPole' in str(env):
        success_rate = np.mean((np.array(steps) >= env._max_episode_steps) * 1.0)
    else:
        raise Exception('Invalid environment name')

    return success_rate

def get_observation_samples(env: gym.Env, n_samples: int) -> np.array:
    """"""
    samples = []
    state = env.reset()
    while len(samples) < n_samples:

        samples.append(np.copy(state))
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        if done:
            state = env.reset()
        else:
            state = next_state

    return np.array(samples)


def set_seed(
    # env,
    seed
):
    """To ensure reproducible runs we fix the seed for different libraries"""
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)

    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # env.seed(seed)
    # gym.spaces.prng.seed(seed)


def get_num_model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
