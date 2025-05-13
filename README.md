# Differentiable-Patterning
A collection of different projects and ideas that use differentiable programming to explore self organising systems with emergent pattern formation. Primarily for my PhD research. 

The general idea is to build auto-differentiable complex systems that can be efficiently trained to yield specified (via data) spatio-temporal patterning.



## Quick start
Clone the repository and install the requirements that best suit you, and play around with the code in the `demo/` directory. The code here is based on the [jax](https://docs.jax.dev/en/latest/) ecosystem, making heavy use of [equinox](https://docs.kidger.site/equinox/), [optax](https://optax.readthedocs.io/en/latest/) and [diffrax](https://docs.kidger.site/diffrax/).

### Requirements 
To run code in this repository, choose:
 - `pip install -r requirements_<DEVICE>.txt`
 - `conda env create -f env_<DEVICE>.yml`
 - `docker pull alexdr1998/jax_equinox_scratch:latest`

Where `<DEVICE>` is either `cpu` or `gpu`, and the docker image is built with cuda


## Code Structure
There are 2 main branches of work here: 
 - NCA (neural cellular automata)
 - PDE (partial differential equations)
 - ~~ABM (agent based modelling)~~

There is considerable overlap between these, so they all inherit from base classes in Common. Everything that can be is a subclass of ```equinox.Module```, so that propagation of gradients works correctly.
### Common structure
- ```Common.model.abstract_model.py``` contains the ```AbstractModel``` class, a subclass of ```equinox.Module```
  - This contains a few extra utility methods like saving to / loading from files
- ```Common.model.spatial_operators.py``` contains the ```Ops``` class, a subclass of ```equinox.Module```
  - This performs finite difference approximations of any 2D vector calculus operations (divergence, gradient, laplacians etc.)
- ```Common.model.boundary.py``` contains the ```model_boundary``` class
  - This enforces complex boundary conditions on model states during training and evalutation
- ```Common.trainer.abstract_data_augmenter_tree.py``` contains a ```DataAugmenterAbstract``` class
  - Subclasses of this handle any data augmentation during training of models.
  - Data should be a PyTree (list) of arrays, which allows for simultaneously training to different sized patterns
- ```Common.trainer.abstract_data_augmenter_array.py``` contains a ```DataAugmenterAbstract``` class
  - Same as above, but only accepts data as one array. Does multi-gpu data parallelism through JAX sharding
- ```Common.trainer.abstract_tensorboard_log.py``` contains a ```Train_log``` class
  - Subclasses of this save various aspects of model parameters/state during and after training
- ```Common.trainer.loss.py``` contains various custom loss functions
- ```Common.utils.py``` contains various helper functions for loading and processing specific datasets

### NCA structure
Everything is subclassed from Common. Important details are that:
- Everything in ```NCA.model.``` is an ```AbstractModel``` subclass that also uses the ```Ops``` class
- ```NCA.trainer.NCA_trainer.py``` includes the ```NCA_Trainer``` class
  - This uses Optax to fit the NCA models to data
- ```NCA.trainer.data_augmenter_*``` Include various ```DataAugmenter``` subclasses, each for training to a different task 
- ```NCA.trainer.tensorboard_log.py``` subclasses ```Train_log``` to visualise model parameters during training
- ```NCA.NCA_visualiser.py``` produces nice plots summarizing model parameters of an ```NCA``` model

### PDE structure
Everything is subclassed from Common. Important details are that:
- ```PDE.model.solver.semidiscrete_solver.py``` contains the ```PDE_solver``` class, a subclass of ```AbstractModel```
  - This uses ```diffrax``` to perform a fully auto-differentiable numerical ODE solve on a spatially discretised PDE (i.e. a system of ODEs)
  - Needs to be initialised with the RHS of the PDE, an ```equinox.Module``` with call signature ```F: t,X,args -> X```  
- ```PDE.model.reaction_diffusion_advection.update.py``` includes an auto-differentiable multi-species reaction diffusion advection equation, parameterised by neural networks
- ```PDE.model.reaction_diffusion_chemotaxis.update.py``` includes an auto-differentiable multi-cell multi-signal reaction diffusion chemotaxis equation, parameterised by neural networks
- ```PDE.model.fixed_models.*``` contains several example PDEs that perform pattern formation
- ```PDE.trainer.PDE_trainer.py``` includes the ```PDE_Trainer``` that uses Optax to fit PDE paramaters such that the solutions of the PDE approximate a given time series
- ```PDE/trainer/optimiser.py``` includes custom ```optax.GradientTransformation()``` that keeps diffusion coefficients non-negative
  
