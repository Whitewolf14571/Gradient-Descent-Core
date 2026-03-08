# Gradient-Descent-Core


# Gradient Descent From Scratch

### Batch vs Stochastic vs Mini-Batch (Machine Learning Optimization Project)

This project implements **three variants of Gradient Descent from scratch using NumPy** and compares their training behavior on a real dataset.

The goal is to understand how different optimization strategies affect **model convergence, training stability, and performance**.

Instead of relying entirely on libraries such as Scikit-learn, the algorithms are implemented manually to develop a deeper intuition about machine learning optimization.

---

# Project Overview

This repository implements and compares the following optimization algorithms:

• **Batch Gradient Descent (BGD)**
• **Stochastic Gradient Descent (SGD)**
• **Mini-Batch Gradient Descent (MBGD)**

The project also includes:

* Feature scaling pipeline
* Model evaluation metrics
* Convergence visualization
* Comparison with **Scikit-learn's SGDRegressor**

---

# Why This Project

Most machine learning tutorials focus on using libraries, but they often hide the underlying training mechanics.

This project focuses on understanding:

* how optimization algorithms update model parameters
* how convergence behavior differs between GD variants
* how feature scaling affects gradient descent
* how custom implementations compare with production ML libraries

---

# Gradient Descent Training Process

Machine learning models learn through an iterative optimization loop.

1. Initialize model parameters
2. Perform forward pass
3. Compute loss function
4. Calculate gradients
5. Update parameters
6. Repeat until convergence

Gradient Descent update rule:

θ = θ − η × gradient

Where

θ → model parameters
η → learning rate

---

# Algorithms Implemented

## Batch Gradient Descent

Uses the **entire dataset** to compute gradients before updating parameters.

Characteristics:

* stable updates
* smooth convergence
* computationally expensive for large datasets

---

## Stochastic Gradient Descent

Updates model parameters using **one sample at a time**.

Characteristics:

* faster updates
* noisy training process
* useful for large datasets

---

## Mini-Batch Gradient Descent

Uses **small batches of samples** during training.

Characteristics:

* balanced stability and speed
* efficient computation
* widely used in deep learning systems

---

# Dataset

A small housing dataset is used for experimentation.

Example structure:

| size | bedrooms | age | target |
| ---- | -------- | --- | ------ |
| 1200 | 3        | 15  | 300000 |
| 1500 | 4        | 10  | 450000 |

Target variable represents **house price**.

---

# Feature Engineering Pipeline

The project includes a simple preprocessing pipeline:

* dataset loading
* feature scaling using **StandardScaler**
* train/test split

Feature scaling is critical because gradient descent is **sensitive to feature magnitude**.

---

# Evaluation Metrics

Model performance is evaluated using:

• **Mean Squared Error (MSE)**
• **Root Mean Squared Error (RMSE)**
• **R² Score**

These metrics help compare the performance of custom implementations against Scikit-learn models.

---

# Experiment

The experiment compares convergence behavior of:

* Batch Gradient Descent
* Stochastic Gradient Descent
* Mini-Batch Gradient Descent

A convergence plot is generated to visualize training behavior.

Example insight:

Batch GD → smooth but slower convergence
SGD → faster but noisy convergence
Mini-Batch GD → balanced performance

The experiment also compares results with:

**Scikit-learn SGDRegressor**

---

# Project Structure

```
gradient-descent-from-scratch

data
  housing.csv

src
  batch_gd.py
  stochastic_gd.py
  minibatch_gd.py
  preprocessing.py
  metrics.py

experiments
  run_experiment.py

plots
  convergence_plot.png

requirements.txt
README.md
```

---

# Installation

Clone repository

```
git clone https://github.com/yourusername/gradient-descent-from-scratch
```

Move into project directory

```
cd gradient-descent-from-scratch
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Run Experiment

Run the main experiment script:

```
python experiments/run_experiment.py
```

This will:

* train all gradient descent models
* compare performance
* generate a convergence plot

---

# Example Output

The script prints evaluation results such as:

Batch GD RMSE
SGD RMSE
MiniBatch GD RMSE
Sklearn SGD RMSE

It also generates a convergence plot showing optimization behavior.

---

# Key Learnings

Through this project:

* Implementing algorithms from scratch builds deeper ML intuition
* Learning rate heavily influences convergence
* SGD introduces noise but improves exploration
* Mini-Batch GD provides a balance between speed and stability
* Feature scaling significantly improves gradient descent performance

---

# Future Improvements

Possible extensions of this project include:

* Momentum optimizer
* Adam optimizer
* Learning rate scheduling
* Loss surface visualization
* GPU-based training experiments

---

# Author

Pankaj Bisht

Machine Learning | MLOPS | AI Systems

---

