% Read the pre-processed feature matrix and labels
X_train = readmatrix('X_train.csv');
Y_train = readmatrix('y_train.csv');

model = py.sklearn.ensemble.RandomForestClassifier(pyargs('n_estimators', int32(100), 'random_state', int32(42)));

% Convert MATLAB matrices to Python objects that fit can work with
X_train_py = py.numpy.array(X_train);
Y_train_py = py.numpy.array(Y_train);

% Train the model
model.fit(X_train_py, Y_train_py);

% Save the model using joblib
py.joblib.dump(model, 'trained_model.pkl');