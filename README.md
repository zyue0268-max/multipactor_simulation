# Multipactor Simulation

This repository contains Python code for secondary electron emission probability calculation and multi-electron multipactor simulation.

The project focuses on modeling electron multiplication in RF structures using secondary electron emission probability models and Monte Carlo particle tracking.

## Files

* furman_probability.py
  Implements the Furman-Pivi secondary electron emission probability model.

* mul_electrons.py
  Runs the multi-electron multipactor simulation and particle tracking process.

## Model Description

The simulation considers secondary electron emission during electron impacts on metal surfaces. The emitted electrons are generated based on probability distributions from the Furman-Pivi model.

Main features include:

* Backscattered electron probability
* Rediffused electron probability
* True secondary electron probability
* Multi-electron generation
* Monte Carlo sampling
* Impact energy calculation
* Electron number evolution during RF cycles

## Requirements

The code requires Python 3 and the following packages:

pip install numpy pandas matplotlib scipy


## How to Run
You do not need to run 'furman_probability.py' separately, because it is only one part of the multi-electron multipactor simulation. The main 
simulation script is 'mul_electrons.py'.

However, if you want to check the probability model separately or see the basic output from the Furman-Pivi model, you can also run 
'furman_probability.py'.

The model used in this code is based on the Furman-Pivi secondary electron emission model. The reference paper link is provided below:

https://doi.org/10.1103/PhysRevSTAB.5.124404
. 


## Results
This code may take some time to run because the current test uses 9 parameter points, and each point is repeated 5 times using Monte Carlo sampling.

After the simulation finishes, the output will be a table containing several key quantities, including `multipactor_ratio`, the fitting quality of the 
logarithmic electron growth curve (`R2`), `avg_final_N`, `avg_alpha`, and the average impact energy.

I would like to clarify that this is only the first draft of the simulation code. The current parameter scan is still very coarse, and the runtime is 
relatively long. At this stage, the main purpose is to test whether the overall simulation workflow can run successfully, including particle tracking, 
secondary electron generation,multipactor judgment, and result collection.

I am currently working on improving the code structure, optimizing the runtime, and refining the parameter scan range for more efficient and accurate 
simulations.


