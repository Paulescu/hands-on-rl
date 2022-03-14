# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['src']

package_data = \
{'': ['*']}

install_requires = \
['gym>=0.21.0,<0.22.0',
 'jupyter>=1.0.0,<2.0.0',
 'jupyterlab>=3.3.0,<4.0.0',
 'matplotlib>=3.5.0,<4.0.0',
 'pandas>=1.3.4,<2.0.0',
 'seaborn>=0.11.2,<0.12.0',
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
    'python_requires': '>=3.7.1,<4.0',
}


setup(**setup_kwargs)
