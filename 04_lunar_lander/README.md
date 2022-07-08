<div align="center">
<h1>Policy Gradients to land on the Moon</h1>
<h3><i>“That's one small step for your gradient ascent, one giant leap for your ML career.”</i></h3>
<h4>-- Pau quoting Neil Armstrong</h4>
</div>

![](http://datamachines.xyz/wp-content/uploads/2022/05/jagoda_and_kai-2048x1536.jpg)

## Table of Contents
* [Welcome 🤗](#welcome-)
* [Lecture transcripts](#lecture-transcripts)
* [Quick setup](#quick-setup)
* [Notebooks](#notebooks)
* [Let's connect](#lets-connect)

## Welcome 🤗

Today we will learn about Policy Gradient methods, and use them to land on the Moon.

Ready, set, go!

## Lecture transcripts

[📝 1. Policy gradients](http://datamachines.xyz/2022/05/06/policy-gradients-in-reinforcement-learning-to-land-on-the-moon-hands-on-course/)  

## Quick setup

Make sure you have Python >= 3.7. Otherwise, update it.

1. Pull the code from GitHub and cd into the `04_lunar_lander` folder:
    ```
    $ git clone https://github.com/Paulescu/hands-on-rl.git
    $ cd hands-on-rl/04_lunar_lander
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

- [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
- [Policy gradients with rewards as weights](notebooks/02_vanilla_policy_gradient_with_rewards_as_weights.ipynb)
- [Policy gradients with rewards-to-go as weights](notebooks/03_vanilla_policy_gradient_with_rewards_to_go_as_weights.ipynb)
- [Homework](notebooks/04_homework.ipynb)

## Let's connect!

Do you wanna become a PRO in Machine Learning?

👉🏽 Subscribe to the [datamachines newsletter](https://datamachines.xyz/subscribe/) 🧠

👉🏽 Follow me on [Twitter](https://twitter.com/paulabartabajo_) and [LinkedIn](https://www.linkedin.com/in/pau-labarta-bajo-4432074b/) 💡
