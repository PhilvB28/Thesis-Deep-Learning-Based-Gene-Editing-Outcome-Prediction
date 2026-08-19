This Project contains the the Deep Learning model Crispr-Caps, a prediction tool for gene editing outcome prediction, based on Capsule Networks. 

## Repository Structure
- datasets – Contains the datasets used for training and evaluating the models.
- models – Contains the different neural network architectures used in the project.
- saved_model – Contains trained model checkpoints.
- utils – Contains utility functions for data processing and sequence encoding.

  main.py – Main entry point for configuring and running model training.
- train.py – Training procedure for the hard parameter-sharing models.
- trainsoft.py – Training procedure for the soft parameter-sharing model.
- test.py – Functions for evaluating trained models and calculating performance metrics.
- PSO.py – Particle Swarm Optimization for hyperparameter optimization.

## Requirements:
- torch: 2.7.0
- scikit-learn: 1.6.1
- pandas: 2.2.3
- numpy: 2.1.3
- pyswarms: 1.3.0
