import os
from typing import List, Optional, Tuple
from pathlib import Path
import json
from pdb import set_trace as stop

from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import gym

from src.model_factory import get_model
from src.utils import (
    get_agent_id,
    set_seed,
    get_num_model_parameters,
    get_logger, get_model_path
)
from src.config import TENSORBOARD_LOG_DIR, SAVED_AGENTS_DIR

def reward_to_go(rews):

    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


class VPGAgent:

    def __init__(
        self,
        env_name: str = 'LunarLander-v2',
        learning_rate: float = 3e-4,
        hidden_layers: List[int] = [32],
        gradient_weights: str = 'rewards'
    ):
        assert gradient_weights in {'rewards', 'rewards-to-go'}

        self.env_name = env_name
        self.env = gym.make(env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.n

        # stochastic policy network
        # the outputs of this network are un-normalized probabilities for each
        # action (aka logits)
        self.policy_net = get_model(input_dim=self.obs_dim,
                                    output_dim=self.act_dim,
                                    hidden_layers=hidden_layers)
        print(f'Policy network with {get_num_model_parameters(self.policy_net):,} parameters')
        print(self.policy_net)


        self.optimizer = Adam(self.policy_net.parameters(), lr=learning_rate)

        self.gradient_weights = gradient_weights

        self.hparams = {
            'learning_rate': learning_rate,
            'hidden_layers': hidden_layers,
            'gradient_weights': gradient_weights,

        }

    def act(self, obs: torch.Tensor):
        """
        Action selection function (outputs int actions, sampled from policy)
        """
        return self._get_policy(obs).sample().item()

    def train(
        self,
        epochs: int = 1000,
        steps_per_epoch: int = 4000,
        logger: Optional[SummaryWriter] = None,
        model_path: Optional[Path] = None,
        seed: Optional[int] = 0,
        freq_eval_in_epochs: Optional[int] = 10,
    ):
        """
        """
        total_steps = 0
        save_model = True if model_path is not None else False
        best_avg_reward = -np.inf

        # fix seeds to ensure reproducible training runs
        set_seed(self.env, seed)

        for i in range(epochs):

            # use current policy to collect trajectories
            states, actions, weights, rewards = self._collect_trajectories(n_samples=steps_per_epoch)

            # one step of gradient ascent to update policy parameters
            loss = self._update_parameters(states, actions, weights)

            # log epoch metrics
            print('epoch: %3d \t loss: %.3f \t reward: %.3f' % (i, loss, np.mean(rewards)))
            if logger is not None:
                # we use total_steps instead of epoch to render all plots in Tensorboard comparable
                # Agents wit different batch_size (aka steps_per_epoch) are fairly compared this way.
                total_steps += steps_per_epoch
                logger.add_scalar('train/loss', loss, total_steps)
                logger.add_scalar('train/episode_reward', np.mean(rewards), total_steps)

            # evaluate the agent on a fixed set of 100 episodes
            if (i + 1) % freq_eval_in_epochs == 0:
                rewards, steps = self.evaluate(n_episodes=100)

                avg_reward = np.mean(rewards)
                if save_model and (avg_reward > best_avg_reward):
                    self.save_to_disk(model_path)
                    print(f'Best model! Average reward = {avg_reward:.2f}')
                    best_avg_reward = avg_reward

    def evaluate(self, n_episodes: Optional[int] = 100, seed: Optional[int] = 1234) -> Tuple[List[float], List[float]]:
        """
        """
        # output metrics
        reward_per_episode = []
        steps_per_episode = []

        # fix seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)

        for i in tqdm(range(0, n_episodes)):

            state = self.env.reset()
            rewards = 0
            steps = 0
            done = False
            while not done:

                action = self.act(torch.as_tensor(state, dtype=torch.float32))

                next_state, reward, done, info = self.env.step(action)

                rewards += reward
                steps += 1
                state = next_state

            reward_per_episode.append(rewards)
            steps_per_episode.append(steps)

        return reward_per_episode, steps_per_episode

    def _collect_trajectories(self, n_samples: int):

        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for reward-to-go weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = self.env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # collect experience by acting in the environment with current policy
        while True:

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            # act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            action = self.act(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = self.env.step(action)

            # save action, reward
            batch_acts.append(action)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                if self.gradient_weights == 'rewards':
                    # the weight for each logprob(a|s) is the total reward for the episode
                    batch_weights += [ep_ret] * ep_len
                elif self.gradient_weights == 'rewards-to-go':
                    # the weight for each logprob(a|s) is the total reward AFTER the action is taken
                    batch_weights += list(reward_to_go(ep_rews))
                else:
                    raise NotImplemented

                # reset episode-specific variables
                obs, done, ep_rews = self.env.reset(), False, []

                # end experience loop if we have enough of it
                if len(batch_obs) > n_samples:
                    break

        return batch_obs, batch_acts, batch_weights, batch_rets

    def _update_parameters(self, states, actions, weights) -> float:
        """
        One step of policy gradient update
        """
        self.optimizer.zero_grad()

        loss = self._compute_loss(
            obs=torch.as_tensor(states, dtype=torch.float32),
            act=torch.as_tensor(actions, dtype=torch.int32),
            weights=torch.as_tensor(weights, dtype=torch.float32)
        )

        # compute gradients
        loss.backward()

        # update parameters with Adam
        self.optimizer.step()

        return loss.item()

    def _compute_loss(self, obs, act, weights):
        logp = self._get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def _get_policy(self, obs):
        """
        Get action distribution given the current policy
        """
        logits = self.policy_net(obs)
        return Categorical(logits=logits)

    @classmethod
    def load_from_disk(cls, env_name: str, path: Path):
        """
        We recover all necessary variables to be able to evaluate the agent.

        NOTE: training state is not stored, so it is not possible to resume
        an interrupted training run as it was.
        """
        # load hyper-params
        with open(path / 'hparams.json', 'r') as f:
            hparams = json.load(f)

        # generate Python object
        agent = cls(env_name, **hparams)

        agent.policy_net = torch.load(path / 'model')
        agent.policy_net.eval()

        return agent

    def save_to_disk(self, path: Path) -> None:
        """"""
        if not path.exists():
            os.makedirs(path)

        # save hyper-parameters in a json file
        with open(path / 'hparams.json', 'w') as f:
            json.dump(self.hparams, f)

        # save main model
        torch.save(self.policy_net, path / 'model')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--gradient_weights', type=str, default='rewards')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument("--hidden_layers", type=int, nargs="+",)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=4000)
    args = parser.parse_args()

    vpg_agent = VPGAgent(
        env_name=args.env,
        gradient_weights=args.gradient_weights,
        learning_rate=args.lr,
        hidden_layers=args.hidden_layers,
    )

    # generate a unique agent_id, that we later use to save results to disk, as
    # well as TensorBoard logs.
    agent_id = get_agent_id(args.env)

    # tensorboard logger to see training curves
    # log_dir = TENSORBOARD_LOG_DIR / args.env / agent_id
    # logger = SummaryWriter(log_dir)
    logger = get_logger(env_name=args.env, agent_id=agent_id)

    # path to save policy network weights
    model_path = get_model_path(env_name=args.env, agent_id=agent_id)

    # start training
    vpg_agent.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        logger=logger,
        model_path=model_path,
    )