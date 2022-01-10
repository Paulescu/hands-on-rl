from typing import Dict
from argparse import ArgumentParser
from pdb import set_trace as stop

import optuna
import gym
import numpy as np
import mlflow

from src.q_agent import QAgent
from src.utils import get_agent_id
from src.config import TENSORBOARD_LOG_DIR, SAVED_AGENTS_DIR, ML_FLOW_EXPERIMENTS
from src.utils import set_seed
from src.loops import train, evaluate


N_EPISODES_TO_EVALUATE = 1000 # 1000


def sample_hyper_parameters(
    trial: optuna.trial.Trial,
    force_linear_model: bool = False,
) -> Dict:

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    discount_factor = trial.suggest_categorical("discount_factor", [0.9, 0.95, 0.99])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    memory_size = trial.suggest_categorical("memory_size", [int(1e4), int(5e4), int(1e5)])

    # we update the main model parameters every 'freq_steps_train' steps
    freq_steps_train = trial.suggest_categorical('freq_steps_train', [1, 4, 8, 16, 128, 256])

    # we update the target model parameters every 'freq_steps_update_target' steps
    freq_steps_update_target = trial.suggest_categorical('freq_steps_update_target', [10, 100, 1000])

    # minimum memory size we want before we start training
    # e.g.    0 --> start training right away.
    # e.g 1,000 --> start training when there are at least 1,000 sample trajectories in the agent's memory
    n_steps_warm_up_memory = trial.suggest_categorical("n_steps_warm_up_memory", [1000, 5000])

    # how many consecutive gradient descent steps to perform when we update the main model parameters
    n_gradient_steps = trial.suggest_categorical("n_gradient_steps", [1, 4, 16])

    # model architecture to approximate q values
    if force_linear_model:
        # linear model
        nn_hidden_layers = None
    else:
        # neural network hidden layers
        nn_hidden_layers = trial.suggest_categorical("nn_hidden_layers", ["None", "[64, 64]", "[256, 256]"])
        nn_hidden_layers = {"None": None, "[64, 64]": [64, 64], "[256, 256]": [256, 256]}[nn_hidden_layers]

    # how large do we let the gradients grow before capping them?
    # Explosive gradients can be an issue and this hyper-parameters helps mitigate it.
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [1, 10])

    # should we scale the inputs before feeding them to the model?
    normalize_state = trial.suggest_categorical('normalize_state', [True, False])

    # start value for the exploration rate
    epsilon_start = trial.suggest_categorical("epsilon_start", [0.9])

    # final value for the exploration rate
    epsilon_end = trial.suggest_uniform("epsilon_end", 0, 0.2)

    # for how many steps do we decrease epsilon from its starting value to
    # its final value `epsilon_end`
    steps_epsilon_decay = trial.suggest_categorical("steps_epsilon_decay", [int(1e3), int(1e4), int(1e5)])

    seed = trial.suggest_int('seed', 0, 2 ** 32 - 1)

    return {
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'batch_size': batch_size,
        'memory_size': memory_size,
        'freq_steps_train': freq_steps_train,
        'freq_steps_update_target': freq_steps_update_target,
        'n_steps_warm_up_memory': n_steps_warm_up_memory,
        'n_gradient_steps': n_gradient_steps,
        'nn_hidden_layers': nn_hidden_layers,
        'max_grad_norm': max_grad_norm,
        'normalize_state': normalize_state,
        'epsilon_start': epsilon_start,
        'epsilon_end': epsilon_end,
        'steps_epsilon_decay': steps_epsilon_decay,
        'seed': seed,
    }


def objective(
    trial: optuna.trial.Trial,
    force_linear_model: bool = False,
    n_episodes_to_train: int = 2000,
):
    env_name = 'CartPole-v1'
    env = gym.make('CartPole-v1')

    with mlflow.start_run():

        agent_id = get_agent_id(env_name)
        mlflow.log_param('agent_id', agent_id)

        # hyper-parameters
        args = sample_hyper_parameters(trial, force_linear_model=force_linear_model)
        mlflow.log_params(trial.params)

        # create agent object
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
            # log_dir=TENSORBOARD_LOG_DIR / env_name / agent_id
        )

        # fix seed before training
        set_seed(args['seed'])
        train(agent, env,
              n_episodes=n_episodes_to_train,
              log_dir=TENSORBOARD_LOG_DIR / env_name / agent_id,
              n_episodes_evaluate_agent=100,
              freq_episodes_evaluate_agent=n_episodes_to_train+1)

        agent.save_to_disk(SAVED_AGENTS_DIR / env_name / agent_id)

        # evaluate its performance
        rewards, steps = evaluate(agent, env,
                                  n_episodes=N_EPISODES_TO_EVALUATE,
                                  epsilon=0.00)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        mlflow.log_metric('mean_reward', mean_reward)
        mlflow.log_metric('std_reward', std_reward)

    return mean_reward


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--trials', type=int, required=True)
    parser.add_argument('--episodes', type=int, required=True)
    parser.add_argument('--force_linear_model', dest='force_linear_model', action='store_true')
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.set_defaults(force_linear_model=False)
    args = parser.parse_args()

    # set Mlflow experiment name
    mlflow.set_experiment(args.experiment_name)

    # set Optuna study
    study = optuna.create_study(study_name=args.experiment_name,
                                direction='maximize',
                                load_if_exists=True)

    # Wrap the objective inside a lambda and call objective inside it
    # Nice trick taken from https://www.kaggle.com/general/261870
    func = lambda trial: objective(trial, force_linear_model=args.force_linear_model)

    # run Optuna
    study.optimize(func, n_trials=args.trials)