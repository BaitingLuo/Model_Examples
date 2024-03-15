% Step 1: Download the dataset and save it to a local file
dataURL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
localFilePath = 'iris.data';
websave(localFilePath, dataURL);

% Step 2: Load and preprocess the data from the local file
fileID = fopen(localFilePath, 'r');
if fileID == -1
    error('File opening failed.');
end
dataCell = textscan(fileID, '%f%f%f%f%s', 'Delimiter', ',', 'EndOfLine', '\n');
fclose(fileID);
features = cell2mat(dataCell(1:4));
labels = dataCell{5};
binaryLabels = double(strcmp(labels, 'Iris-versicolor'));

% Step 3: Train the model
mdl = fitclinear(features, binaryLabels, 'Learner', 'logistic');

% Step 4: Save the trained model
save('logisticRegressionModel.mat', 'mdl');