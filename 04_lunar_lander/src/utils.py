import os
from typing import Callable, Dict, Tuple, List
import pathlib
from pathlib import Path
import json
from pdb import set_trace as stop

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.config import TENSORBOARD_LOG_DIR, SAVED_AGENTS_DIR


# def snake_to_camel(word):
#     import re
#     return ''.join(x.capitalize() or '_' for x in word.split('_'))

def get_agent_id(env_name: str) -> str:
    """"""
    dir = Path(SAVED_AGENTS_DIR) / env_name
    if not dir.exists():
        os.makedirs(dir)

    ids = []
    for id in os.listdir(dir):
        try:
            ids.append(int(id))
        except:
            pass
    if len(ids) > 0:
        agent_id = max(ids) + 1
    else:
        agent_id = 0
    # stop()

    return str(agent_id)


def set_seed(
    env,
    seed
):
    """To ensure reproducible runs we fix the seed for different libraries"""
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    env.seed(seed)
    env.action_space.seed(seed)

    import torch
    torch.manual_seed(seed)

    # Deterministic operations for CuDNN, it may impact performances
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_num_model_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_logger(env_name: str, agent_id: str) -> SummaryWriter:
    return SummaryWriter(TENSORBOARD_LOG_DIR / env_name / agent_id)

def get_model_path(env_name: str, agent_id: str) -> Path:
    """
    Returns path where we save train artifacts, including:
     -> the policy network weights
     -> json with hyperparameters
     """
    return SAVED_AGENTS_DIR / env_name / agent_id


