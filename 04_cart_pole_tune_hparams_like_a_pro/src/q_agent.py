"""
"""
from pathlib import Path
from typing import Union, Callable, Tuple, List
import random
from argparse import ArgumentParser
import uuid
from pdb import set_trace as stop

import gym
import numpy as np
import torch
# import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import neptune.new as neptune

from src.model_factory import get_model
from src.agent_memory import AgentMemory
from src.utils import (
    load_env_config,
    get_input_output_dims,
    get_epsilon_decay_fn,
    # load_default_hyperparameters,
    # get_observation_samples,
    set_seed,
    get_num_model_parameters
)
from src.loops import train
from src.config import LOCAL_MODEL_CHECKPOINTS_DIR

# load environment variables into Python dict
config = load_env_config()

class QAgent:

    def __init__(
        self,
        env: gym.Env = None,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        batch_size: int = 64,
        memory_size: int = 10000,
        freq_steps_update_target: int = 1000,
        n_steps_warm_up_memory: int = 1000,
        freq_steps_train: int = 16,
        n_gradient_steps: int = 8,
        nn_hidden_layers: List[int] = None,
        max_grad_norm: int = 10,
        normalize_state: bool = False,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        steps_epsilon_decay: float = 50000,

        run = None,
    ):
        """
        :param env:
        :param learning_rate: size of the updates in the SGD/Adam formula
        :param discount_factor: discount factor for future rewards
        :param batch_size: number of (s,a,r,s') experiences we use in each SGD
        update
        :param memory_size: number of experiences the agent keeps in the replay
        memory
        :param freq_steps_update_target: frequency at which we copy the
        parameter
        from the main model to the target model.
        :param n_steps_warm_up_memory: number of experiences we require to have
        in memory before we start training the agent.
        :param freq_steps_train: frequency at which we update the main model
        parameters
        :param n_gradient_steps: number of SGD/Adam updates we perform when we
        train the main model.
        :param nn_hidden_layers: architecture of the main and target models.
        :param max_grad_norm: used to clipped gradients if they become too
        large.
        :param normalize_state: True/False depending if you want to normalize
        the raw states before feeding them into the model.
        :param epsilon_start: starting exploration rate
        :param epsilon_end: final exploration rate
        :param steps_epsilon_decay: number of step in which epsilon decays from
        'epsilon_start' to 'epsilon_end'
        :param log_dir: Tensorboard logging folder
        """
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
        # Adam optimizer is a safe and standard choice
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
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

        # Neptune AI logger
        self.run = run

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

        # copy parameters from the `self.q_net` to `self.target_q_net`
        self._copy_params_to_target_q_net()

    def _copy_params_to_target_q_net(self):
        """
        Copies parameters from q_net to target_q_net
        """
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)

    def _preprocess_state(self, state: np.array) -> np.array:

        if len(state.shape) == 1:
            # add extra dimension to make sure it is 2D
            s = state.reshape(1, -1)
        else:
            s = state

        return s

    def act(self, state: np.array, epsilon: float = None) -> int:
        """
        Behavioural policy
        """
        if epsilon is None:
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
        s = torch.from_numpy(s).float().to(self.device)
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
            done_batch = torch.tensor(batch.done).float().to(self.device)

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

        if self.run is not None:
            self.run['train/loss'].log(np.mean(losses))

    def save_checkpoint(self) -> None:
        """"""
        if self.run is not None:
            local_checkpoint_file = str(LOCAL_MODEL_CHECKPOINTS_DIR / str(uuid.uuid4()))
            torch.save(self.q_net, local_checkpoint_file)
            self.run['model_checkpoints/q_net'].upload(local_checkpoint_file)

    @classmethod
    def load_checkpoint(cls, run_id: str):
        """
        Returns the trained agent from the given 'run_id' in Neptune.
        """
        raise Exception('TODO')

        # generate Python object
        agent = cls()

        # agent.q_net = torch.load(path / 'model')
        # agent.q_net.eval()
        #
        return agent



def parse_arguments():
    """
    Hyper-parameters are set either from command line or from the `hyperparameters.yaml' file.
    Parameters set throught the command line have priority over the default ones
    set in the yaml file.
    """

    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True)

    # parameteric q learning hyper-parameters
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--discount_factor', type=float)
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
    parser.add_argument("--max_grad_norm", type=float, default=10)
    parser.add_argument('--seed', type=int, default=0)

    # general train/eval parameters
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--n_episodes_evaluate_agent', type=int, default=100)
    parser.add_argument('--freq_episodes_evaluate_agent', type=int, default=100)

    args = parser.parse_args()

    # save args inside Python dict
    args_dict = {}
    for arg in vars(args):
        args_dict[arg] = getattr(args, arg)

    print('Hyper-parameters')
    for key, value in args_dict.items():
        print(f'{key}: {value}')

    return args_dict


if __name__ == '__main__':

    args = parse_arguments()

    # start Neptune logger
    run = neptune.init(
        project=config['NEPTUNE_PROJECT'],
        api_token=config['NEPTUNE_API_TOKEN'],
    )

    # log all the parameters for this run
    run["parameters"] = args

    # setup the environment
    env = gym.make(args['env'])

    # fix seeds to ensure reproducibility between runs
    set_seed(env, args['seed'])

    agent = QAgent(
        env,

        # hyper-parameters
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

        run=run, # Neptune run for logging purposes
    )

    train(
        agent,
        env,
        n_episodes=args['episodes'],
        n_episodes_evaluate_agent=args['n_episodes_evaluate_agent'],
        freq_episodes_evaluate_agent=args['freq_episodes_evaluate_agent'],
        run=run,
    )

    run.stop()