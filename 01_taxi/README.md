<div align="center">
<h1>Q-learning to drive a taxi 🚕</h1>
<h3><i>You talkin' to me?</i></h3>
<h4>-- Robert de Niro (Taxi driver)</h4>
</div>

<figure>
<img src="http://datamachines.xyz/wp-content/uploads/2021/11/pexels-helena-jankovic%CC%8Cova%CC%81-kova%CC%81c%CC%8Cova%CC%81-5870314.jpg" style="width:100%">
<figcaption align = "center">Venice’s taxis 👆 by <a href="https://www.pexels.com/@helen1?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels">Helena Jankovičová Kováčová</a> from Pexels 🙏</figcaption>
</figure>

## Table of Contents
* [Welcome 🤗](#welcome-)
* [Quick setup](#quick-setup)
* [Lecture transcripts](#lecture-transcripts)
* [Notebooks](#notebooks)
* [Let's connect](#lets-connect)

## Welcome 🤗
This is part 1 of the Hands-on RL course.

Let's use (tabular) Q-learning to teach an agent to solve the [Taxi-v3](https://gym.openai.com/envs/Taxi-v3/) environment
from OpenAI gym.

Fasten your seat belt and get ready. We are ready to depart!


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
    $ (venv) python setup.py develop
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


## Lectures transcripts

[📝 Q learning](http://datamachines.xyz/2021/12/06/hands-on-reinforcement-learning-course-part-2-q-learning/)  


## Notebooks

- [Explore the environment](notebooks/00_environment.ipynb)
- [Random agent baseline](notebooks/01_random_agent_baseline.ipynb)
- [Q-agent](notebooks/02_q_agent.ipynb)
- [Hyperparameter tuning](notebooks/03_q_agent_hyperparameters_analysis.ipynb)
- [Homework](notebooks/04_homework.ipynb)

## Let's connect!

Do you wanna become a PRO in Machine Learning?

👉🏽 Subscribe to the [datamachines newsletter](https://datamachines.xyz/subscribe/).

👉🏽 Follow me on [Medium](https://pau-labarta-bajo.medium.com/).





