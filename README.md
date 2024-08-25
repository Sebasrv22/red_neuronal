# Neural Network Implementation in NumPy and TensorFlow

## Introduction

This project demonstrates two implementations of a simple feedforward neural network for binary classification using synthetic data generated in the shape of two circles. The first implementation is done using NumPy, and the second implementation utilizes TensorFlow and Keras. The goal is to compare and contrast the performance and simplicity of implementing a neural network with both libraries.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Documentation](#documentation)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Installation

To run the code, you need to have Python installed along with the required libraries. You can install the necessary dependencies using pip:

pip install numpy scipy tensorflow keras matplotlib 

## Usage

NumPy Implementation
The NumPy code is a manual implementation of a neural network with a focus on understanding the underlying mechanics of forward and backward propagation. To run the NumPy implementation:

Ensure all dependencies are installed.
Run the script in your Python environment:

python numpy_neural_network.py

## TensorFlow Implementation
The TensorFlow code leverages the Keras API for simplicity and ease of use. This implementation demonstrates the power of high-level libraries in building and training neural networks. To run the TensorFlow implementation:

Ensure all dependencies are installed.
Run the script in your Python environment:
python tensorflow_neural_network.py


## Features

Synthetic Data Generation: Creates two distinct datasets in the form of circles, representing two different classes.
Custom Activation Functions: Implements ReLU and Sigmoid activation functions from scratch in the NumPy version.
Manual Backpropagation: Includes the manual computation of gradients and weight updates in the NumPy version.
Keras Sequential API: Utilizes the high-level Keras API to define, compile, and train the neural network in TensorFlow.

## Dependencies

Python 3.x
NumPy
SciPy
TensorFlow
Keras (included with TensorFlow)
Matplotlib (for potential data visualization)
Configuration

No special configuration is needed. The scripts are self-contained and should run out of the box after dependencies are installed.

## Documentation

NumPy Implementation
Activation Functions: Implements ReLU and Sigmoid functions along with their derivatives.
Layer Class: Defines a neural network layer with weights and biases.
Training: Performs forward propagation, computes loss using Mean Squared Error, and updates weights using backpropagation.
TensorFlow Implementation
Keras Model: Defines a simple Sequential model with three layers.
Compilation: Uses Stochastic Gradient Descent (SGD) as the optimizer and Mean Squared Error (MSE) as the loss function.
Training and Validation: Splits the data into training, validation, and test sets, and trains the model for 500 epochs.

## Examples

Examples of usage are embedded within the scripts. After running the scripts, you will see output logs that provide insight into the training process, including loss values and final predictions for both datasets.

## Troubleshooting

Ensure that you have the correct versions of all dependencies installed.
If you encounter memory issues, try reducing the dataset size or the number of epochs.
## Contributors

Sebastián Romero - Initial work

## License

This project is licensed under the MIT License - see the LICENSE file for details.
