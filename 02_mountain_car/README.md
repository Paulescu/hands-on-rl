# SARSA to get the mountain car out of the whole 🚃

> *Be like a train; go in the rain, go in the sun, go in the storm, go in the dark tunnels! Be like a train; concentrate on your road and go with no hesitation!*
>
> Mehmet Murat Ildan

This is part 2 of my course Hands-on reinforcement learning.

In this part we use SARSA to help a poor car win the battle against gravity!


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

5. and activate the virtual environment
    ```bash
    $ poetry shell
    ```

### Notebooks

1. [Explore the environment](notebooks/00_environment.ipynb)
2. [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
3. [SARSA agent](notebooks/02_sarsa_agent.ipynb)
4. [Momentum agent](notebooks/03_momentum_agent_baseline.ipynb)
5. [Homework](notebooks/04_homework.ipynb)