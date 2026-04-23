# MDM2 Group 10
---
__This Github repository contains all code used to generate images and results in our project. Below is a breakdown of each folder :__

The `Hongze_File` folder contains the code used in order to generate results for the evolution model.
* The summary PDF fully describes the implementation including parameter values which we didn't have space to include in the technical note
* `dye_evolution_model.ipynb` contains the main functions for the evolution model with explanations of their implementation

The `WB_files` folder contains the code relating to the Hamiltonian Monte Carlo method of optimisation.
* The full diagnostic plots are shown in the `outputs` folder
* `Evolution_model.py` and `Optical_model.py` are simply copies of the code in evolution and optical model notebooks from Lin and Qusai's section respectively converted into python files so functions could be imported
* `hmc_inference_v2.py` is the main HMC implementation and `hmc_diagnostics.py` runs the same process on two chains along with diagnostics to evaluate the performance

The `qusais_files` folder contains the code for the creation of the optical model.
* `SimpleModal_Optical.ipynb` holds the updated optical model which is fully differentiable via PyTorch autograd allowing it to seamlessly fit into the HMC code

The `zachs_files` folder contains the code for the gradient descent method of optimisation.

The `manim.py` file contains the code for creating some of the animations on our presentation.
