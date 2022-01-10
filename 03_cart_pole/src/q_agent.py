"""
We use PyTorch for all agents:

- Linear model trained one sample at a time -> Easy to train, slow and results are not great.
- Linear model trained with batches of data. -> Faster to train, but results are still not good.
- NN trained with batches -> Promising, but it looks like it does not train..
- NN with memory buffer -> Fix sample autocorrelation
- NN with memory buffer and target network for stability. -> RL trick (called double Q-learning)

"""
import os
from pathlib import Path
from typing import Union, Callable, Tuple, List
import random
from argparse import ArgumentParser
import json
from pdb import set_trace as stop

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import joblib

from src.model_factory import get_model
from src.agent_memory import AgentMemory
from src.utils import (
    get_agent_id,
    get_input_output_dims,
    get_epsilon_decay_fn,
    load_default_hyperparameters,
    get_observation_samples,
    set_seed,
    get_num_model_parameters
)
from src.loops import train
from src.config import TENSORBOARD_LOG_DIR, SAVED_AGENTS_DIR


class QAgent:

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        batch_size: int = 64,
        memory_size: int = 10000,
        freq_steps_update_target: int = 1000,
        n_steps_warm_up_memory: int = 1000,
        freq_steps_train: int = 16,
        n_gradient_steps: int = 8,
        nn_hidden_layers: List[int] = [256, 256],
        max_grad_norm: int = 10,
        normalize_state: bool = False,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        steps_epsilon_decay: float = 50000,
        log_dir: str = None,
    ):
        # TODO: call super() method
        self.env = env

        # general hyper-parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # replay memory we use to sample experiences and update parameters
        # `memory_size` defines the maximum number of past experiences we want the
        # agent remember.
        self.memory_size = memory_size
        self.memory = AgentMemory(memory_size)

        # number of experiences we take at once from `self.memory` to update parameters
        self.batch_size = batch_size

        # hyper-parameters to control exploration of the environment
        self.steps_epsilon_decay = steps_epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_fn = get_epsilon_decay_fn(epsilon_start, epsilon_end, steps_epsilon_decay)
        self.epsilon = None

        # create q model(s). Plural because we use 2 models: main one, and the other for the target.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net, self.target_q_net = None, None
        self._init_models(nn_hidden_layers)
        print(f'{get_num_model_parameters(self.q_net):,} parameters')
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate) # Adam optimizer is a safe and standard choice
        self.max_grad_norm = max_grad_norm

        # hyper-parameters to control how often or when we do certain things, like
        # - update the main net parameters
        self.freq_steps_train = freq_steps_train
        # - update the target net parameters
        self.freq_steps_update_target = freq_steps_update_target
        # - start training until the memory is big enough
        assert n_steps_warm_up_memory > batch_size, 'batch_size must be larger than n_steps_warm_up_memory'
        self.n_steps_warm_up_memory = n_steps_warm_up_memory
        # - number of gradient steps we perform every time we update the main net parameters
        self.n_gradient_steps = n_gradient_steps

        # state variable we use to keep track of the number of calls to `observe()`
        self._step_counter = 0

        # input normalizer
        self.normalize_state = normalize_state
        self.state_normalizer = None
        if normalize_state:
            state_samples = get_observation_samples(env, n_samples=500000)
            self.max_states = state_samples.max(axis=0)
            self.min_states = state_samples.min(axis=0)
            # print('Max states: ', self.max_states)
            # print('Min states: ', self.min_states)

        # create a tensorboard logger if `log_dir` was provided
        # logging becomes crucial to understand what is not working in our code.
        self.log_dir = log_dir
        if log_dir:
            self.logger = SummaryWriter(log_dir)

        # save hyper-parameters
        self.hparams = {
            'learning_rate': learning_rate,
            'discount_factor': discount_factor,
            'batch_size': batch_size,
            'memory_size': memory_size,
            'freq_steps_update_target': freq_steps_update_target,
            'n_steps_warm_up_memory': n_steps_warm_up_memory,
            'freq_steps_train': freq_steps_train,
            'n_gradient_steps': n_gradient_steps,
            'nn_hidden_layers': nn_hidden_layers,
            'max_grad_norm': max_grad_norm,
            'normalize_state': normalize_state,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'steps_epsilon_decay': steps_epsilon_decay,
        }

    def _init_models(self, nn_hidden_layers):

        # state is a vector of dimension 4, and 2 are the possible actions
        input_dim, output_dim = get_input_output_dims(str(self.env))
        self.q_net = get_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=nn_hidden_layers,
        )
        self.q_net.to(self.device)

        # target q-net
        self.target_q_net = get_model(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=nn_hidden_layers,
        )
        self.target_q_net.to(self.device)

        # copy parameters from the `self.q_net`
        self._copy_params_to_target_q_net()

    def _copy_params_to_target_q_net(self):
        """
        Copies parameters from q_net to target_q_net
        """
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

    def _normalize_state(self, state: np.array) -> np.array:
        """"""
        return (state - self.min_states) / (self.max_states - self.min_states)

    def _preprocess_state(self, state: np.array) -> np.array:

        # state = np.copy(state_)

        if len(state.shape) == 1:
            # add extra dimension to make sure it is 2D
            s = state.reshape(1, -1)
        else:
            s = state

        if self.normalize_state:
            s = self._normalize_state(s)

        return s

    def act(self, state: np.array, epsilon: float = None) -> int:
        """
        Behavioural policy
        """
        if not epsilon:
            # update epsilon
            self.epsilon = self.epsilon_fn(self._step_counter)
            epsilon = self.epsilon

        if random.uniform(0, 1) < epsilon:
            # Explore action space
            action = self.env.action_space.sample()
            return action

        # make sure s is a numpy array with 2 dimensions,
        # and normalize it if `self.normalize_state = True`
        s = self._preprocess_state(state)

        # forward pass through the net to compute q-values for the 3 actions
        s = torch.from_numpy(s).float()
        q_values = self.q_net(s)

        # extract index max q-value and reshape tensor to dimensions (1, 1)
        action = q_values.max(1)[1].view(1, 1)

        # tensor to float
        action = action.item()

        return action

    def observe(self, state, action, reward, next_state, done) -> None:

        # preprocess state
        s = self._preprocess_state(state)
        ns = self._preprocess_state(next_state)

        # store new experience in the agent's memory.
        self.memory.push(s, action, reward, ns, done)

        self._step_counter += 1

    def replay(self) -> None:

        if self._step_counter % self.freq_steps_train != 0:
            # update parameters every `self.freq_steps_update_target`
            # this way we add inertia to the agent actions, as they are more sticky
            return

        if self._step_counter < self.n_steps_warm_up_memory:
            # memory needs to be larger, no training yet
            return

        if self._step_counter % self.freq_steps_update_target == 0:
            # we update the target network parameters
            # self.target_nn.load_state_dict(self.nn.state_dict())
            self._copy_params_to_target_q_net()

        losses = []
        for i in range(0, self.n_gradient_steps):

            # get batch of experiences from the agent's memory.
            batch = self.memory.sample(self.batch_size)

            # A bit of plumbing to transform numpy arrays to PyTorch tensors
            state_batch = torch.cat([torch.from_numpy(s).float().view(1, -1) for s in batch.state]).to(self.device)
            action_batch = torch.cat([torch.tensor([[a]]).long().view(1, -1) for a in batch.action]).to(self.device)
            reward_batch = torch.cat([torch.tensor([r]).float() for r in batch.reward]).to(self.device)
            next_state_batch = torch.cat([torch.from_numpy(s).float().view(1, -1) for s in batch.next_state]).to(self.device)
            done_batch = torch.tensor(batch.done).float()

            # q_values for all 3 actions
            q_values = self.q_net(state_batch)

            # keep only q_value for the chosen action in the trajectory, i.e. `action_batch`
            q_values = q_values.gather(1, action_batch)

            with torch.no_grad():
                # q-values for each action in next_state
                next_q_values = self.target_q_net(next_state_batch)

                # extract max q-value for each next_state
                next_q_values, _ = next_q_values.max(dim=1)

                # TD target
                target_q_values = (1 - done_batch) * next_q_values * self.discount_factor + reward_batch

            # compute loss
            # loss = F.mse_loss(q_values, target_q_values.unsqueeze(1))
            loss = F.mse_loss(q_values.squeeze(1), target_q_values)
            losses.append(loss.item())

            # backward step to adjust network parameters
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
            self.optimizer.step()

        if self.log_dir:
            self.logger.add_scalar('train/loss', np.mean(losses), self._step_counter)

    def save_to_disk(self, path: Path) -> None:
        """"""
        if not path.exists():
            os.makedirs(path)

        # save hyper-parameters in a json file
        with open(path / 'hparams.json', 'w') as f:
            json.dump(self.hparams, f)

        if self.normalize_state:
            np.save(path / 'max_states.npy', self.max_states)
            np.save(path / 'min_states.npy', self.min_states)

        # save main model
        torch.save(self.q_net, path / 'model')

    @classmethod
    def load_from_disk(cls, env: gym.Env, path: Path):
        """
        We recover all necessary variables to be able to evaluate the agent.

        NOTE: training state is not stored, so it is not possible to resume
        an interrupted training run as it was.
        """
        # load hyper-params
        with open(path / 'hparams.json', 'r') as f:
            hparams = json.load(f)

        # generate Python object
        agent = cls(env, **hparams)

        agent.normalize_state = hparams['normalize_state']
        if hparams['normalize_state']:
            # load max/min states to normalize the input data to the model
            agent.max_states = np.load(path / 'max_states.npy')
            agent.min_states = np.load(path / 'min_states.npy')

        agent.q_net = torch.load(path / 'model')
        agent.q_net.eval()

        return agent



def parse_arguments():
    """
    Hyper-parameters are set either from command line or from the `hyperparameters.yaml' file.
    Parameters set throught the command line have priority over the default ones
    set in the yaml file.
    """

    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--discount_factor', type=float)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--epsilon_start', type=float)
    parser.add_argument('--epsilon_end', type=float)
    parser.add_argument('--steps_epsilon_decay', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--memory_size', type=int)
    parser.add_argument('--n_steps_warm_up_memory', type=int)
    parser.add_argument('--freq_steps_update_target', type=int)
    parser.add_argument('--freq_steps_train', type=int)
    parser.add_argument('--normalize_state', dest='normalize_state', action='store_true')
    parser.set_defaults(normalize_state=None)
    parser.add_argument('--n_gradient_steps', type=int,)
    parser.add_argument("--nn_hidden_layers", type=int, nargs="+",)
    parser.add_argument('--nn_init_method', type=str, default='default')
    parser.add_argument('--loss', type=str)
    parser.add_argument("--max_grad_norm", type=float, default=10)
    parser.add_argument('--n_episodes_evaluate_agent', type=int, default=100)
    parser.add_argument('--freq_episodes_evaluate_agent', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    assert args.env in {'MountainCar-v0', 'CartPole-v1'}

    args_dict = load_default_hyperparameters(args.env)

    for arg in vars(args):
        # overwrite default values with command line values, if provided
        command_line_value = getattr(args, arg)
        if command_line_value is not None:
            args_dict[arg] = command_line_value

    print('Hyper-parameters')
    for key, value in args_dict.items():
        print(f'{key}: {value}')

    return args_dict


if __name__ == '__main__':

    args = parse_arguments()

    env = gym.make(args['env'])

    # to ensure reproducibility between runs we need to fix a few seeds.
    set_seed(env, args['seed'])

    # this is the name we use to save the agent and its tensorboard logs.
    agent_id = get_agent_id(args['env'])
    print('agent_id: ', agent_id)

    agent = QAgent(
        env,
        learning_rate=args['learning_rate'],
        discount_factor=args['discount_factor'],
        batch_size=args['batch_size'],
        memory_size=args['memory_size'],
        freq_steps_train=args['freq_steps_train'],
        freq_steps_update_target=args['freq_steps_update_target'],
        n_steps_warm_up_memory=args['n_steps_warm_up_memory'],
        n_gradient_steps=args['n_gradient_steps'],
        nn_hidden_layers=args['nn_hidden_layers'],
        max_grad_norm=args['max_grad_norm'],
        normalize_state=args['normalize_state'],
        epsilon_start=args['epsilon_start'],
        epsilon_end=args['epsilon_end'],
        steps_epsilon_decay=args['steps_epsilon_decay'],
        log_dir=TENSORBOARD_LOG_DIR / args['env'] / agent_id
    )
    agent.save_to_disk(SAVED_AGENTS_DIR / args['env'] / agent_id)

    try:
        train(agent, env,
              n_episodes=args['episodes'],
              log_dir=TENSORBOARD_LOG_DIR / args['env'] / agent_id,
              n_episodes_evaluate_agent=args['n_episodes_evaluate_agent'],
              freq_episodes_evaluate_agent=args['freq_episodes_evaluate_agent'],
              # max_steps=args['max_steps']
              )

        agent.save_to_disk(SAVED_AGENTS_DIR / args['env'] / agent_id)
        print(f'Agent {agent_id} was saved')

    except KeyboardInterrupt:
        # save the agent before quitting...
        agent.save_to_disk(SAVED_AGENTS_DIR / args['env'] / agent_id)
        print(f'Agent {agent_id} was saved')