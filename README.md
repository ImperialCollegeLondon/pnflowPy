# pnflowPy - classical pore network flow simulation (python version)

This repository hosts the classical network flow simulation code written in python called pnflowPy. 
This code was written, not as an attempt to re-write pnflow in python, but to have a simplified version that could be easily extended to simulate other flow phenomena. Thus, the code is not as robust and fast as pnflow though could be further developed later. 

# Getting Started
  - Install your favorite Python distribution if you don't already have one (we used Anaconda). You may want to choose a distribution that has the standard packages pre-installed
  - Install required libraries (check the Dependencies below)
  - Download the code
  - Run the examples.

# Dependencies
There are a few extra libraries to install:
  - pandas
  - scipy
  - sortedcontainers
  - numpy_indexed
  - petsc4py

# Installation
Quick installation can be achieved by replicating the environment in Anaconda:
  1. Clone the repo
     ```python
     git clone https://github.com/ImperialCollegeLondon/pnflowPy.git
  2. Configure conda
     ```python
     conda update conda
     conda config --set ssl_verify false
  3. Replicate the environment:
     ```python
     conda env create -f environment.yml
  4. Activate the environment:
     ```python
     conda activate flowmodel

# Running pnflowPy
You can run the pnflowPy by running the following command in terminal or in Windows command-prompt (cmd):
    ```python
    python main.py input_pnflow.dat
This will work if pnflow is in system PATH. If otherwise, type the full path to the pnflow.



# Contributions
It would be very great to make any contributions to this project to make it better. If you would like to contribute, please fork the repo and create a pull request or simply open an issue with the tag "contribution". Please give this project a star. Thank you!

# License
Distributed under the MIT License. See LICENSE.txt for more information.

# Contact and References
For contacts and references please see: https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling

Alternatively, contact Ademola Adebimpe:

  - Email: a.adebimpe21@imperial.ac.uk
  - Additional Email: ai.bimpe@gmail.com
