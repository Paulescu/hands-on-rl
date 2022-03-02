<div align="center">
<h1>Parametric Q learning to keep the balance</h1>
<h2>Let's (perfectly) solve the CartPole</h2>
</div>

> *There exists everywhere a medium in things, determined by equilibrium.*
>
> --_Dmitri Mendeleev_

## Table of Contents

* [Quick setup](#setup)
* [Notebooks](#generate-training-data-and-train-model)
* [Spread the word]()

## Hi there 👋

It's Pau, and this is problem number 3 of my course Hands-on RL course.

In today's lecture we enter new territory...

A territory where function approximation (aka supervised machine learning)
meets good old Reinforcement Learning.

And this is how Deep RL is born.

We will solve the Cart Pole environment of OpenAI using **parametric Q-learning**.

Today's lesson is split into 3 parts:

👉 [Parametric Q-learning](http://datamachines.xyz/2022/01/18/hands-on-reinforcement-learning-course-part-4-parametric-q-learning/)  
👉 [Deep Q learning](http://datamachines.xyz/2022/02/11/hands-on-reinforcement-learning-course-part-5-deep-q-learning/)  
👉 [Hyper-parameter tuning]()


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

### 2. Notebooks

0. [Explore the environment](notebooks/00_environment.ipynb)
1. [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
2. [Linear Q agent with bad hyper-parameters](notebooks/02_linear_q_agent_bad_hyperparameters.ipynb)
3. [Linear Q agent with good hyper-parameters](notebooks/03_linear_q_agent_good_hyperparameters.ipynb)
4. [Homework](notebooks/04_homework.ipynb)


6. [Crash course on neural networks](notebooks/05_crash_course_on_neural_nets.ipynb)
7. [Deep Q agent with bad hyper-parameters](notebooks/06_deep_q_agent_bad_hyperparameters.ipynb)
8. [Deep Q agent with good hyper-parameters](notebooks/07_deep_q_agent_good_hyperparameters.ipynb)
9. [Homework](notebooks/08_homework.ipynb)


10. [Hyperparameter search](notebooks/09_hyperparameter_search.ipynb)
11. [Homework](notebooks/10_homework.ipynb)