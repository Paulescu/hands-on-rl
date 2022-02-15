import os
import pathlib
root_dir = pathlib.Path(__file__).parent.resolve().parent

LOCAL_MODEL_CHECKPOINTS_DIR = root_dir / 'model_checkpoints'

if not LOCAL_MODEL_CHECKPOINTS_DIR.exists():
    os.makedirs(LOCAL_MODEL_CHECKPOINTS_DIR)