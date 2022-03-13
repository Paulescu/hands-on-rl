# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['src']

package_data = \
{'': ['*']}

install_requires = \
['PyYAML>=6.0,<7.0',
 'gdown>=4.2.0,<5.0.0',
 'gym>=0.21.0,<0.22.0',
 'jupyter>=1.0.0,<2.0.0',
 'matplotlib>=3.5.0,<4.0.0',
 'mlflow>=1.22.0,<2.0.0',
 'numpy>=1.21.4,<2.0.0',
 'optuna>=2.10.0,<3.0.0',
 'pandas>=1.3.5,<2.0.0',
 'pyglet>=1.5.21,<2.0.0',
 'pyngrok>=5.1.0,<6.0.0',
 'sklearn>=0.0,<0.1',
 'tensorboard>=2.7.0,<3.0.0',
 'torch>=1.10.1,<2.0.0',
 'tqdm>=4.62.3,<5.0.0']

setup_kwargs = {
    'name': 'src',
    'version': '0.1.0',
    'description': '',
    'long_description': None,
    'author': 'Pau',
    'author_email': 'plabartabajo@gmail.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.7.1,<3.8',
}


setup(**setup_kwargs)
