# Simulaton
Fluid Simulation using HPC

# Installation
For a CPU-only installation, clone the repository, then pip install ..

For GPU support, you'll need to first install the version of jaxlib appropriate for your python, CUDA, and CuDNN versions. See the jaxlib install instructions for more details.

## Starting point
This is the starting point of every simulation

![](Results/Images/vlcsnap-00001.png) 

The following image represents the velocity vectors from different points in the simulation

![](Results/Images/vlcsnap-00004.png)

## Mid-Simulation 
This is the mid-point of the simulation, according to the particular forces acting upon this simulation

![](Results/Images/vlcsnap-00002.png) 

## End-Simulation
This is the ending point, namely the 420th frame for our simulation purpose. 

![](Results/Images/vlcsnap-00003.png) 

You can change the frames from within the Fluid_Sim.ipynb along with other properties like viscosity, frame-rate, resolution **{(720 x 720) in our case}**.

The code is self-explanatory and there are comments in appropriate places for helping readability.
