clear
close all

x = linspace(0, 2, 1001); % High-resolution input
y = sin(x * pi);          % True sine values
y = max(y,-0.5);          % occationally binding constraint

% Normalize x to the range [-1, 1]
x_norm = normalize(x, 'range', [-1, 1]);

% Create a feedforward network with more capacity
net = feedforwardnet([20 20]); % Two hidden layers
%net.trainFcn = 'traingd';   % Stochastic Gradient Descent
net.trainParam.lr = 0.001;  % Learning rate
net.trainParam.epochs = 1000;          % Train for more epochs
net.layers{1}.transferFcn   = 'tansig';  % Use tansig for the first layer
net.layers{2}.transferFcn   = 'purelin'; % Use purelin for the second layer
net.layers{end}.transferFcn = 'purelin';

% Train the network
net = train(net, x_norm, y);

% Simulate the network
y_pred = net(x_norm);

% Plot results
plot(x, y, 'b', 'DisplayName', 'True');
hold on;
plot(x, y_pred, 'r--', 'DisplayName', 'Predicted');
legend;
title('Comparison of True and Predicted Values');
xlabel('x');
ylabel('y');
