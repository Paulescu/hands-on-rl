<div align="center">
<h1>Parametric Q learning to keep the balance</h1>
<h2>Let's (perfectly) solve the CartPole</h2>
</div>

> *There exists everywhere a medium in things, determined by equilibrium.*
>
> --_Dmitri Mendeleev_

![](http://datamachines.xyz/wp-content/uploads/2022/01/pexels-yogendra-singh-1701202.jpg)

## Welcome back 🤗!

In today's lecture we enter new territory...

A territory where function approximation (aka supervised machine learning)
meets good old Reinforcement Learning.

And this is how Deep RL is born.

We will solve the Cart Pole environment of OpenAI using **parametric Q-learning**.

Today's lesson is split into 3 parts.

👉 <a href="http://datamachines.xyz/2022/01/18/hands-on-reinforcement-learning-course-part-4-parametric-q-learning/" target="_blank">Parametric Q learning</a>  
👉 [Deep Q learning](http://datamachines.xyz/2022/02/11/hands-on-reinforcement-learning-course-part-5-deep-q-learning/)  
👉 [Hyperparameter search](http://datamachines.xyz/2022/03/03/hyperparameters-in-deep-rl-hands-on-course/)

## Quick setup

Make sure you have Python >= 3.7. Otherwise, update it.

1. Pull the code from GitHub and cd into the `03_cart_pole` folder:
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
