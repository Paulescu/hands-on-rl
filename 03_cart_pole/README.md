# Linear Q learning to keep the balance 💃

This is part 3 of my course Hands-on reinforcement learning.

Today we enter new territory...

A territory where function approximation (aka supervised machine learning)
meets good old Reinforcement Learning.

> *There exists everywhere a medium in things, determined by equilibrium.*
>
> --_Dmitri Mendeleev_

There is a lot to diggest here, so I decided to split this lecture in 3 parts:

### Table of Contents

* [Setup](#setup)
* [Notebooks](#notebooks)


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
3. [Linear Q agent with bad hyper-parameters](notebooks/02_linear_q_agent_bad_hyperparameters.ipynb)
4. [Linear Q agent with good hyper-parameters](notebooks/03_linear_q_agent_good_hyperparameters.ipynb)
5. [Homework](notebooks/04_homework.ipynb)