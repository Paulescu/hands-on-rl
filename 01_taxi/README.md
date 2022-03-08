# Q-learning to drive a taxi 🚕
> *You talkin' to me?*
>
> Robert de Niro (Taxi driver)

This is part 1 of my course Hands-on reinforcement learning.

In this part we use Q-learning to teach an agent to drive a taxi.

## Quick setup

Make sure you have Python >= 3.7. Otherwise, update it.

1. Pull the code from GitHub and cd into the `01_taxi` folder:
    ```
    $ git clone https://github.com/Paulescu/hands-on-rl.git
    $ cd hands-on-rl/01_taxi
    ```

2. Create a virtual environment and activate it.
    ```
    $ virtualenv -p python3 venv
    $ source venv/bin/activate
    ```

    From this point onwards commands run inside the  virtual environment.


3. Install dependencies and code from `src` folder (in editable mode `-e`, so you can experiment with the code)
    ```
    $ (venv) pip install -e .
    ```

4. Open the notebooks, either with good old Jupyter
    ```
    $ (venv) jupyter notebook
    ```
    or Jupyterlab
    ```
    $ (venv) jupyter lab
    ```

5. Play and learn. And do the homework 😉.

## Notebooks

1. [Explore the environment](notebooks/00_environment.ipynb)
2. [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
3. [Q-agent](notebooks/02_q_agent.ipynb)
4. [Hyper-parameter tuning](notebooks/03_q_agent_hyperparameters_analysis.ipynb)
5. [Homework](notebooks/04_homework.ipynb)





