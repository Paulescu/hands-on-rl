import os
import pathlib
root_dir = pathlib.Path(__file__).parent.resolve().parent

SAVED_AGENTS_DIR = root_dir / 'saved_agents'
TENSORBOARD_LOG_DIR = root_dir / 'tensorboard_logs'
OPTUNA_STUDIES = root_dir / 'sqlite:///optuna.db'

if not SAVED_AGENTS_DIR.exists():
    os.makedirs(SAVED_AGENTS_DIR)

if not TENSORBOARD_LOG_DIR.exists():
    os.makedirs(TENSORBOARD_LOG_DIR)

