import os
from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = [line for line in f if line and line[0] not in "#-"]

setup(
    name="src",
    version=os.getenv("PACKAGE_VERSION") or "dev",
    url="TBD",
    author="pau.labarta.bajo",
    author_email="plabartabajor@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Private :: Do Not Upload",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
)