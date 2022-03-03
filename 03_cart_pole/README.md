<div align="center">
<h1>Parametric Q learning to keep the balance</h1>
<h2>Let's (perfectly) solve the CartPole</h2>
</div>

> *There exists everywhere a medium in things, determined by equilibrium.*
>
> --_Dmitri Mendeleev_

![](http://datamachines.xyz/wp-content/uploads/2022/01/pexels-yogendra-singh-1701202.jpg)

## Welcome 🤗

This is Pau,

and this is problem number 3 of my course Hands-on RL course.

In today's lecture we enter new territory...

A territory where function approximation (aka supervised machine learning)
meets good old Reinforcement Learning.

And this is how Deep RL is born.

We will solve the Cart Pole environment of OpenAI using **parametric Q-learning**.

Today's lesson is split into 3 parts.

👉 <a href="http://datamachines.xyz/2022/01/18/hands-on-reinforcement-learning-course-part-4-parametric-q-learning/" target="_blank">Parametric Q learning</a>  
👉 [Deep Q learning](http://datamachines.xyz/2022/02/11/hands-on-reinforcement-learning-course-part-5-deep-q-learning/)  
👉 [Hyperparameter search]()


## Quick setup

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

5. Activate the virtual environment
    ```bash
    $ poetry shell
    ```

6. Set PYTHONPATH and launch jupyter (jupyter-lab param may fix launch problems on some systems)
    ```bash
    $ export PYTHONPATH=".."
    $ jupyter-lab --NotebookApp.use_redirect_file=False
    ```

All `python` commands are executed froms this virtual environment created by poetry.

## Notebooks

Parametric Q-learning
- [Explore the environment](notebooks/00_environment.ipynb)
- [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
- [Linear Q agent with bad hyper-parameters](notebooks/02_linear_q_agent_bad_hyperparameters.ipynb)
- [Linear Q agent with good hyper-parameters](notebooks/03_linear_q_agent_good_hyperparameters.ipynb)
- [Homework](notebooks/04_homework.ipynb)

Deep Q-learning
- [Crash course on neural networks](notebooks/05_crash_course_on_neural_nets.ipynb)
- [Deep Q agent with bad hyper-parameters](notebooks/06_deep_q_agent_bad_hyperparameters.ipynb)
- [Deep Q agent with good hyper-parameters](notebooks/07_deep_q_agent_good_hyperparameters.ipynb)
- [Homework](notebooks/08_homework.ipynb)

Hyperparameter search
- [Hyperparameter search](notebooks/09_hyperparameter_search.ipynb)
- [Homework](notebooks/10_homework.ipynb)
