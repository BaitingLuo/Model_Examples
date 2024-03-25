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

When saving a logistic regression model for future use, such as in a `.pickle` file, it's crucial to ensure the model exposes certain functionalities to enable both retraining and straightforward predictions. This ensures seamless integration and flexibility in various operational environments. The essential functions to be exposed are:

### `fit(self, X, y)`
- **Purpose:** Retrain or update the model with new data. This function should take new feature matrices (`X`) and target vectors (`y`) to adjust the model's parameters based on additional data or to retrain from scratch.
- **Parameters:**
  - `X`: Feature matrix of the new data.
  - `y`: Target vector corresponding to `X`.

### `predict(self, X)`
- **Purpose:** Make predictions using the trained model. Given a feature matrix (`X`), this function should return the predicted labels, allowing for the model's application in making decisions or further analyses.
- **Parameters:**
  - `X`: Feature matrix of the data for which predictions are desired.

### `predict_proba(self, X)`
- **Optional but Recommended:** Provides the probability estimates for each class, offering more detailed insight than binary predictions, especially in applications where understanding the confidence level of predictions is crucial.
- **Parameters:**
  - `X`: Feature matrix of the data for which probability estimates are desired.

### `save(self, filepath)`
- **Purpose:** Efficiently save the model to disk, using a format like pickle, for future use. This method encapsulates the serialization process, abstracting it away from the end user.
- **Parameters:**
  - `filepath`: Destination path where the model should be saved.

### `load(filepath)`
- **Purpose:** Static method to load a previously saved model. This function ensures that a model can be quickly reloaded without needing to retrain from scratch, facilitating ease of use in production environments.
- **Parameters:**
  - `filepath`: Path to the saved model file.

Ensuring these functions are well-defined and accessible in the logistic regression model's class definition will enable users to seamlessly transition between training, prediction, and retraining phases, enhancing the model's utility and flexibility.


