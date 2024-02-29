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
     conda activate flowenv

  If numba was not installed, try:
      ```python
      conda install -c numba numba
  
  If numpy-indexed was not installed, try:
      ```python
      pip install numpy-indexed


# Running pnflowPy
To run the pnflowPy , you should first generate the networks from the images using the **[pnextract](https://github.com/ImperialCollegeLondon/pnextract)** or **[poreXtractor](https://github.com/ImperialCollegeLondon/poreOccupancyAnalysis)** module, see the documentation of pnextract executable. Then you should copy the generated networks into the data folder and edit the input_pnflow.dat by setting the NETWORK keyword and other keywords. Full descriptions of the keywords in the input_pnflow.dat can be found in doc/pnflow_guide.pdf inside the **[pnflow](https://github.com/ImperialCollegeLondon/pnflow)** module. However, only the keywords inside the input_pnflow.dat have been implemented, others might be added later for more robustness.

You can then run the pnflowPy by doing the following:
  1.  Change directory into the pnflowPy folder.
  2.  Running the following command in terminal or in Windows command-prompt (cmd):
      ```python
      python main.py data/input_pnflow.dat

The generated network files of a 1000 cube water-wet Bentheimer sandstone have been provided in the data folder (image can be found in https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/) and this could be used for an example.

# Contributions
It would be very great to make any contributions to this project to make it better. If you would like to contribute, please fork the repo and create a pull request or simply open an issue with the tag "contribution". Please give this project a star. Thank you!

# License
Distributed under the MIT License.

# Contact and References
For contacts and references please see: https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling

Alternatively, contact Ademola Adebimpe:

  - Email: **a.adebimpe21@imperial.ac.uk**
  - Additional Email: **ai.bimpe@gmail.com**
