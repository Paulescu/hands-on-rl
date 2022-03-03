# Define SAVED_AGENTS_DIR and create dir if missing
import os
import pathlib
root_dir = pathlib.Path(__file__).parent.resolve().parent
SAVED_AGENTS_DIR = root_dir / 'saved_agents'
os.makedirs(SAVED_AGENTS_DIR, exist_ok=True)
