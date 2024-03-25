# Iris Classification Model

## Overview

This project involves a MATLAB script designed to classify iris flowers into the 'Iris-versicolor' species using logistic regression. The process is divided into four main steps: downloading the dataset, loading and preprocessing the data, training the model, and saving the trained model.

## Prerequisites

- MATLAB R2015b or later
- Internet connection for downloading the dataset

## Step-by-Step Guide

### Step 1: Download the Dataset

The Iris dataset is downloaded from the UCI Machine Learning Repository and saved locally.


% Download the dataset and save it to a local file
dataURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
localFilePath = 'iris.data';
websave(localFilePath, dataURL);

### Step 2: Load and Preprocess the Data

The data is loaded from the local file, and preprocessing involves extracting features and converting labels into binary format.
% Load and preprocess the data from the local file
fileID = fopen(localFilePath, 'r');
if fileID == -1
    error('File opening failed.');
end
dataCell = textscan(fileID, '%f%f%f%f%s', 'Delimiter', ',', 'EndOfLine', '\n');
fclose(fileID);
features = cell2mat(dataCell(1:4));
labels = dataCell{5};
binaryLabels = double(strcmp(labels, 'Iris-versicolor'));

### Step 3: Train the Model
A logistic regression model is trained using the preprocessed features and labels.
% Train the model
mdl = fitclinear(features, binaryLabels, 'Learner', 'logistic');

### Step 4: Save the Trained Model
The trained logistic regression model is saved to a file for future use.
% Save the trained model
save('logisticRegressionModel.mat', 'mdl');
