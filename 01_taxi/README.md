# Q-learning to drive a taxi 🚕
> *You talkin' to me?*
>
> Robert de Niro (Taxi driver)

This is part 1 of my course Hands-on reinforcement learning.

In this part we use Q-learning to teach an agent to drive a taxi.

### Quick setup

The easiest way to get the code working in your machine is by using [Poetry](https://python-poetry.org/docs/#installation).


1. You can install Poetry with this one-liner:
    ```bash
    $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
    ```

2. Git clone the code
    ```bash
    $ git clone https://github.com/Paulescu/hands-on-rl.git 
    ```

3. Navigate to this lesson code `01_taxi`
    ```bash
    $ cd hands-on-rl/01_taxi
    ```

4. Install all dependencies from `pyproject.toml:
    ```bash
    $ poetry install
    ```

5. Activate the virtual environment
    ```bash
    $ poetry shell
    ```

6. Set PYTHONPATH and launch jupyter
    ```bash
    $ export PYTHONPATH=".."
    $ jupyter-lab
    ```

### Notebooks

1. [Explore the environment](notebooks/00_environment.ipynb)
2. [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
3. [Q-agent](notebooks/02_q_agent.ipynb)
4. [Hyper-parameter tuning](notebooks/03_q_agent_hyperparameters_analysis.ipynb)
5. [Homework](notebooks/04_homework.ipynb)





