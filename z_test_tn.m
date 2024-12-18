clear
close all

% Define the network: reluLayer, leakyReluLayer, sigmoidLayer, tanhLayer, eluLayer, softmaxLayer
layers = [
    sequenceInputLayer(1)
    fullyConnectedLayer(20)       % More neurons
    reluLayer                     % Smoother activation
    fullyConnectedLayer(20)       % Additional hidden layer
    reluLayer
    fullyConnectedLayer(1)
    regressionLayer
];

% Training options sgdm, adam, etc.
options = trainingOptions('adam', ...
    'MiniBatchSize', 128, ...
    'MaxEpochs', 1000, ...
    'InitialLearnRate', 0.01, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

sample = 5001;

% Training data
x_train = linspace(0, 2, sample)'; % Input
y_train = sin(x_train * pi);       % Target
y_train = max(y_train,-0.5);

% Reshape input for sequenceInputLayer
x_train = reshape(x_train, [1, sample, 1]); % [numFeatures x numTimesteps x numObservations]
y_train = reshape(y_train, [1, sample, 1]); % Reshape target to match x dimensions

% Normalize inputs
x_train_norm = normalize(x_train, 'range', [-1, 1]);

% Train the network
trainedNetwork = trainNetwork(x_train_norm, y_train, layers, options);

% Test data
x_test = linspace(0, 2, sample)'; % Test input
x_test_norm = normalize(x_test, 'range', [-1, 1]);
x_test_norm = reshape(x_test_norm, [1, sample, 1]); % [numFeatures x numTimesteps x numObservations]

% Predict using the trained network
y_pred = predict(trainedNetwork, x_test_norm);

% Plot results
plot(x_test, y_train, 'b', 'DisplayName', 'True sin(x)');
hold on;
plot(x_test, y_pred, 'r--', 'DisplayName', 'Predicted sin(x)');
legend;
title('True vs Predicted sin(x) using Trained Network');
xlabel('x');
ylabel('y');
