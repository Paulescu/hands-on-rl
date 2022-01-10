# Function approximation to keep the balance 💃
👉 [Read in datamachines]()
👉 [Read in Towards Data Science]()


This is part 3 of my course Hands-on reinforcement learning.

In this part we enter new territory.

A territory where function approximation (aka supervised machine learning)
meets good old Reinforcement Learning.

> *There exists everywhere a medium in things, determined by equilibrium.*
>
> --_Dmitri Mendeleev_

### Table of Contents

* [Setup](#setup)
* [Notebooks](#notebooks)
* [Command line interface](#model-inference-as-a-rest-api)
* [Results](#results)


### Setup

The easiest way to get the code working in your machine is by using [Poetry](https://python-poetry.org/docs/#installation).


1. You can install Poetry with this one-liner:
    ```bash
    $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    ```

2. Git clone the code
    ```bash
    $ git clone https://github.com/Paulescu/hands-on-rl.git 
    ```

3. Navigate to this lesson code `03_cart_pole`
    ```bash
    $ cd hands-on-rl/03_cart_pole
    ```

4. Install all dependencies:
    ```bash
    $ poetry install
    ```

5. and activate the virtual environment
    ```bash
    $ poetry shell
    ```

All `python` commands are executed froms this virtual environment created by poetry.

### Notebooks

1. [Explore the environment](notebooks/00_environment.ipynb)
2. [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
3. [Linear Q agent](notebooks/02_linear_q_agent.ipynb)
4. [Deep Q agent](notebooks/03_deep_q_agent.ipynb)
5. [Hyper-parameter optimization]()
6. [Homework](notebooks/04_homework.ipynb)


## TODOs

CartPole
- [x] NN to solve the problem perfectly.
  - [x] Fix bug global_steps
  - [x] Save hparams
  - [x] Save preprocessor
  - [x] Train agent with preprocessing of state.
  - [x] agent.load_from_disk()

- [ ] Integrate Optuna to find best hyper-parameters for linear agent.
- [ ] Integrate Optuna to find best hyper-parameters for deep agent.



- [ ] NN model notebook where we train and evaluate the agent  
- [ ] Generate video from trained agent.

- [ ] Linear model notebook where we train and evaluate the agent
- [ ] Generate video from trained agent.

