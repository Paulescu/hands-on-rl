<div align="center">
<h1>Parametric Q learning to solve the Cart Pole</h1>
<h3><i>There exists everywhere a medium in things, determined by equilibrium.</i></h3>
<h4>-- Dmitri Mendeleev</h4>
</div>

![](http://datamachines.xyz/wp-content/uploads/2022/01/pexels-yogendra-singh-1701202.jpg)

## Table of Contents
* [Welcome 🤗](#welcome-)
* [Lecture transcripts](#lecture-transcripts)
* [Quick setup](#quick-setup)
* [Notebooks](#notebooks)
* [Let's connect](#lets-connect)

## Welcome 🤗

In today's lecture we enter new territory...

A territory where function approximation (aka supervised machine learning)
meets good old Reinforcement Learning.

And this is how Deep RL is born.

We will solve the Cart Pole environment of OpenAI using **parametric Q-learning**.

Today's lesson is split into 3 parts.

## Lecture transcripts

[📝 1. Parametric Q learning](http://datamachines.xyz/2022/01/18/hands-on-reinforcement-learning-course-part-4-parametric-q-learning)  
[📝 2. Deep Q learning](http://datamachines.xyz/2022/02/11/hands-on-reinforcement-learning-course-part-5-deep-q-learning/)  
[📝 3. Hyperparameter search](http://datamachines.xyz/2022/03/03/hyperparameters-in-deep-rl-hands-on-course/)

## Quick setup

Make sure you have Python >= 3.7. Otherwise, update it.

1. Pull the code from GitHub and cd into the `01_taxi` folder:
    ```
    $ git clone https://github.com/Paulescu/hands-on-rl.git
    $ cd hands-on-rl/01_taxi
    ```

2. Make sure you have the `virtualenv` tool in your Python installation
    ```
   $ pip3 install virtualenv
   ```

3. Create a virtual environment and activate it.
    ```
    $ virtualenv -p python3 venv
    $ source venv/bin/activate
    ```

   From this point onwards commands run inside the  virtual environment.


3. Install dependencies and code from `src` folder in editable mode, so you can experiment with the code.
    ```
    $ (venv) pip install -r requirements.txt
    $ (venv) export PYTHONPATH="."
    ```

4. Open the notebooks, either with good old Jupyter or Jupyter lab
    ```
    $ (venv) jupyter notebook
    ```
    ```
    $ (venv) jupyter lab
    ```
   If both launch commands fail, try these:
    ```
    $ (venv) jupyter notebook --NotebookApp.use_redirect_file=False
    ```
    ```
    $ (venv) jupyter lab --NotebookApp.use_redirect_file=False
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

## Let's connect!

Do you wanna become a PRO in Machine Learning?

👉🏽 Subscribe to the [datamachines newsletter](https://datamachines.xyz/subscribe/).

👉🏽 Follow me on [Medium](https://pau-labarta-bajo.medium.com/).