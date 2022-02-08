from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
from pdb import set_trace as stop

import zipfile
import gdown
from tqdm import tqdm
import pandas as pd
import gym
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.model_factory import get_model
from src.utils import set_seed
from src.loops import evaluate
from src.q_agent import QAgent
from src.config import DATA_SUPERVISED_ML, SAVED_AGENTS_DIR, TENSORBOARD_LOG_DIR


global_train_step = 0
global_val_step = 0


def download_agent_parameters() -> Path:
    """
    Downloads the agent parameters and hyper-parameters that I trained on my machine
    Returns the path to the unzipped folder.
    """
    # download .zip file from public google drive
    url = 'https://docs.google.com/uc?export=download&id=1KH4ANx84PMmCY6H4FoUnkBLVC1z1A6W6'
    output = SAVED_AGENTS_DIR / 'CartPole-v1' / 'gdrive_agent.zip'
    gdown.download(url, str(output))

    # unzip it
    with zipfile.ZipFile(str(output), "r") as zip_ref:
        zip_ref.extractall(str(SAVED_AGENTS_DIR / 'CartPole-v1'))

    return SAVED_AGENTS_DIR / 'CartPole-v1' / '79'


def simulate_episode(env, agent) -> List[Dict]:
    """"""
    done = False
    state = env.reset()
    samples = []
    while not done:

        action = agent.act(state, epsilon=0.0)
        samples.append({
            's0': state[0],
            's1': state[1],
            's2': state[2],
            's3': state[3],
            'action': action
        })
        state, reward, done, info = env.step(action)

    return samples


def generate_state_action_data(
    env: gym.Env,
    agent: QAgent,
    n_samples: int,
    path: Path
) -> None:
    """"""
    samples = []
    with tqdm(total=n_samples) as pbar:
        while len(samples) < n_samples:
            new_samples = simulate_episode(env, agent)
            pbar.update(len(new_samples))
            samples += new_samples

    pd.DataFrame(samples).to_csv(path, index=False)


class OptimalPolicyDataset(Dataset):

    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X.iloc[idx].values, self.y.iloc[idx]


def get_tensorboard_writer(run_name: str):

    from torch.utils.tensorboard import SummaryWriter
    from src.config import TENSORBOARD_LOG_DIR
    tensorboard_writer = SummaryWriter(TENSORBOARD_LOG_DIR / 'sml' / run_name)
    return tensorboard_writer


def get_train_val_loop(
    model: nn.Module,
    criterion,
    optimizer,
    tensorboard_writer,
):
    global global_train_step, global_val_step
    global_train_step = 0
    global_val_step = 0

    def train_val_loop(
        is_train: bool,
        dataloader: DataLoader,
        epoch: int,
    ):
        """"""
        global global_train_step, global_val_step

        n_batches = 0
        running_loss = 0
        n_samples = 0
        n_correct_predictions = 0

        pbar = tqdm(dataloader)
        for data in pbar:

            # extract batch of features and target values (aka labels)
            inputs, labels = data

            if is_train:
                # zero the parameter gradients
                optimizer.zero_grad()

            # forward
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)

            if is_train:
                # backward + optimize
                loss.backward()
                optimizer.step()

            predicted_labels = torch.argmax(outputs, 1)
            batch_accuracy = (predicted_labels == labels).numpy().mean()

            n_batches += 1
            running_loss += loss.item()
            avg_loss = running_loss / n_batches

            n_correct_predictions += (predicted_labels == labels).numpy().sum()
            n_samples += len(labels)
            avg_accuracy = n_correct_predictions / n_samples
            pbar.set_description(f'Epoch {epoch} - loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}')

            # log to tensorboard
            if is_train:
                global_train_step += 1
                tensorboard_writer.add_scalar('train/loss', loss.item(), global_train_step)
                tensorboard_writer.add_scalar('train/accuracy', batch_accuracy, global_train_step)
                # print('sent logs to TB')
            else:
                global_val_step += 1
                tensorboard_writer.add_scalar('val/loss', loss.item(), global_val_step)
                tensorboard_writer.add_scalar('val/accuracy', batch_accuracy, global_val_step)

    return train_val_loop



def run(
    n_samples_train: int,
    n_samples_test: int,
    hidden_layers: Union[Tuple[int], None],
    n_epochs: int,
):
    env = gym.make('CartPole-v1')

    print('Downloading agent data from GDrive...')
    path_to_agent_data = download_agent_parameters()
    # path_to_agent_data = Path('/Users/paulabartabajo/src/online-courses/hands-on-rl/03_cart_pole/saved_agents/CartPole-v1/79')
    agent = QAgent.load_from_disk(env, path=path_to_agent_data)

    set_seed(env, 1234)
    print('Sanity checking that our agent is really that good...')
    rewards, steps = evaluate(agent, env, n_episodes=100, epsilon=0.0)
    print('Avg reward evaluation: ', np.mean(rewards))

    print('Generating train data for our supervised ML problem...')
    path_to_train_data = DATA_SUPERVISED_ML / 'train.csv'
    env.seed(0)
    generate_state_action_data(env, agent, n_samples=n_samples_train, path=path_to_train_data)

    print('Generating test data for our supervised ML problem...')
    path_to_test_data = DATA_SUPERVISED_ML / 'test.csv'
    env.seed(1)
    generate_state_action_data(env, agent, n_samples=n_samples_test, path=path_to_test_data)

    # load data from disk
    print('Loading CSV files into dataframes...')
    train_data = pd.read_csv(path_to_train_data)
    test_data = pd.read_csv(path_to_test_data)

    # split features and labels
    X_train = train_data[['s0', 's1', 's2', 's3']]
    y_train = train_data['action']
    X_test = test_data[['s0', 's1', 's2', 's3']]
    y_test = test_data['action']

    # PyTorch datasets
    train_dataset = OptimalPolicyDataset(X_train, y_train)
    test_dataset = OptimalPolicyDataset(X_test, y_test)

    batch_size = 64

    # PyTorch dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model architecture
    model = get_model(input_dim=4, output_dim=2, hidden_layers=hidden_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # optimization method
    optimizer = optim.Adam(model.parameters()) #, lr=3e-4)

    import time
    ts = int(time.time())
    tensorboard_writer = SummaryWriter(TENSORBOARD_LOG_DIR / 'sml' / str(ts))
    train_val_loop = get_train_val_loop(model, criterion, optimizer, tensorboard_writer)

    # training loop, with evaluation at the end of each epoch
    # n_epochs = 20
    for epoch in range(n_epochs):
        # train
        train_val_loop(is_train=True, dataloader=train_dataloader, epoch=epoch)

        with torch.no_grad():
            # validate
            train_val_loop(is_train=False, dataloader=test_dataloader, epoch=epoch)

        print('----------')


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--n_samples_train', type=int, default=1000)
    parser.add_argument('--n_samples_test', type=int, default=1000)
    parser.add_argument("--hidden_layers", type=int, nargs="+",)
    parser.add_argument('--n_epochs', type=int, default=20)
    args = parser.parse_args()

    run(n_samples_train=args.n_samples_train,
        n_samples_test=args.n_samples_test,
        hidden_layers=args.hidden_layers,
        n_epochs=args.n_epochs)