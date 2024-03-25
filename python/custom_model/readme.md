# Custom Logistic Regression Model

## Overview

This project features a Python implementation of a logistic regression model from scratch, applied to a binary classification version of the Iris dataset. The logistic regression algorithm is used to predict the probability that a given input point belongs to a certain class. The implementation covers data preprocessing, model training, prediction making, and evaluation of the model's accuracy.

## Prerequisites

- Python 3
- NumPy
- Pandas
- scikit-learn

## Components

### CustomLogisticRegression Class

The `CustomLogisticRegression` class encapsulates the logistic regression algorithm, including methods for fitting the model to the data, predicting probabilities, and classifying new examples.

#### `__init__(self, learning_rate=0.01, n_iterations=1000)`

Initializes the model with a specified learning rate and number of iterations for the gradient descent optimization.

#### `_sigmoid(self, z)`

Private method implementing the sigmoid function, which maps the input 'z' to a value between 0 and 1, representing the probability of the positive class.

#### `fit(self, X, y)`

Trains the logistic regression model using gradient descent. It updates the model's weights based on the training data 'X' and the target values 'y'.

#### `predict_proba(self, X)`

Predicts the probability that each example in 'X' belongs to the positive class.

#### `predict(self, X)`

Classifies each example in 'X' as belonging to the positive class (1) or the negative class (0), based on the predicted probabilities.

### Data Preparation and Model Evaluation

- Data is loaded and split into training and testing sets.
- Feature scaling is applied to standardize the feature values.
- The custom logistic regression model is instantiated, trained on the training set, and used to make predictions on the test set.
- Model accuracy is evaluated by comparing the predicted class labels against the true labels of the test set.

## Usage

To run the logistic regression model:

1. Ensure all prerequisites are installed.
2. Place your binary-class version of the Iris dataset in the same directory as the script and name it `iris_binary.csv`.
3. Execute the script. The output will include the model's accuracy on the test set.

## Abstract Model Requirements

For implementing a logistic regression model, the essential components include:

- **Sigmoid Function:** Maps any real-valued number to the (0, 1) interval, essential for predicting probabilities.
- **Cost Function:** Measures the performance of the model during training, guiding the gradient descent optimization.
- **Gradient Descent:** Optimization algorithm used to minimize the cost function by iteratively adjusting the model's parameters (weights and bias).
- **Prediction Threshold:** A probability threshold (commonly set to 0.5) used to classify predictions into binary classes.

These components work together to enable the logistic regression model to learn from the training data and make accurate predictions.

