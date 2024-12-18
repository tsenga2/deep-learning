
clear
close all

%% Parameters

beta  = 0.96;   % Discount factor
sigma = 2;      % Risk aversion parameter
alpha = 0.3;    % Capital share
delta = 0.1;    % Depreciation rate
rho_z = 0.9;    % TFP persistence
sig_z = 0.02;   % Standard deviation of TFP shocks

params = [beta;sigma;alpha;delta;rho_z;sig_z];
save('text_files\params.txt','params','-ascii','-double');

%% Steady state

k_ss = ( 1/alpha*(1/beta-1+delta) )^(1/(alpha-1));
c_ss = k_ss^alpha - delta*k_ss;
y_ss = k_ss^alpha;
i_ss = delta*k_ss;
r_ss = alpha * y_ss/k_ss;
z_ss = 0;

ss_values = [c_ss;k_ss;y_ss;i_ss;r_ss;z_ss];
save('text_files\ss_values.txt','ss_values','-ascii','-double');

%% Simulate using dynare

dynare model_rbc

% Random simulation using dynare_sim
rng("default")
rng(1)
t_max  = 5000;
t_drop = 500;
shocks = normrnd(0,1,[M_.exo_nbr,t_max]);
for i=1:M_.exo_nbr
    shocks(i,:) = shocks(i,:) * sqrt(M_.Sigma_e(i,i));
end
save('shocks.mat','shocks')

[sim_dev,sim_values,sim_perdev] = dynare_sim(oo_,M_,shocks);
for i=1:M_.endo_nbr
    eval(['sim_',deblank(M_.endo_names{i,1}),' = sim_values(i,:);']);
end

%% Create a feedforward network

net = feedforwardnet([20]);  % Number of nodes in each hidden layer
%net.trainFcn = 'traingd';      % traingdx
%net.trainParam.lr = 0.001;      % Learning rate
%net.trainParam.epochs = 1000;   % Number of epochs
% Transfer functions: purelin,tansig,logsig,poslin(relu),satlin,satlins,etc...
%net.layers{1}.transferFcn   = 'tansig'; % Use tansig for the first layer
%net.layers{2}.transferFcn   = 'poslin'; % Use tansig for the second layer
%net.layers{3}.transferFcn   = 'tansig'; % Use tansig for the third layer
%net.layers{end}.transferFcn = 'purelin'; % Use purelin for the output layer
%net.trainParam.showWindow=0; % Default: 1

%% Initial training

% Each input is each row. Num of variables by Num of observations.
seq_state = [sim_k(t_drop:t_max-1);sim_z(t_drop+1:t_max)];
seq_rhsee = (sim_c(t_drop+1:t_max)).^(-sigma);

init_net = train(net,seq_state,seq_rhsee);

ffn_rhsee = zeros(1,t_max);
ffn_c     = zeros(1,t_max);
ffn_rhsee(2:t_max) = init_net([sim_k(1:t_max-1);sim_z(2:t_max)]);
ffn_c(2:t_max)     = ffn_rhsee(2:t_max).^(-1/sigma);

%% Plot

t_plot = 100;

figure
plot(2:t_plot,sim_c(2:t_plot),'b','LineWidth',3);
hold on
plot(2:t_plot,ffn_c(2:t_plot),'r--','LineWidth',3);

%% Irreversible Investment Model

%mean_k = mean(seq_k);
%min_k  = min(seq_k);
%max_k  = max(seq_k);
%mean_rhsee = mean(seq_rhsee);
%min_rhsee  = min(seq_rhsee);
%max_rhsee  = max(seq_rhsee);

% normalize variables
%seq_k     = (seq_k     - mean_k    )/(max_k    -min_k    );
%seq_rhsee = (seq_rhsee - mean_rhsee)/(max_rhsee-min_rhsee);

% Gauss-Hermite quadrature for integration
Qn  = 5; % Number of nodes
N   = 1; % Number of shocks
vcv = 1; % Var-CoVar matrix of shocks
[nodes,eps_nodes,wgt_nodes] = GH_Quadrature(Qn,N,vcv);

pea_net = init_net;

%% PEA starts

iter_max = 1000;
tol      = 1e-3;   % convergence criterion

t_max  = 1000;
t_drop = 500;

rhsee    = zeros(1,t_max);
rhsee_dr = zeros(1,t_max);
c  = zeros(1,t_max);
k  = zeros(1,t_max);
y  = zeros(1,t_max);
iv = zeros(1,t_max);
z  = sim_z; % use the sequence of shocks generated by dynare.

for iter=1:iter_max 

% initial states for simulation
k(1) = sim_k(1);

% simulation
for t=2:t_max

    rhsee(t) = pea_net([k(t-1);z(t)]);
    c(t)     = rhsee(t)^(-1/sigma);
    k(t)     = exp(z(t))*k(t-1)^alpha + (1-delta)*k(t-1) - c(t);
%    if k(t) < (1-delta)*k(t-1)
%        k(t) = (1-delta)*k(t-1);
%        c(t) = exp(z(t))*k(t-1)^alpha + (1-delta)*k(t-1) - k(t);
%    end

    rhsee_dr(t) = 0;
    for n=1:nodes

        z_nx     = rho_z*z(t) + sig_z*eps_nodes(n);
        rhsee_nx = pea_net([k(t);z_nx]);
        c_nx     = rhsee_nx^(-1/sigma);
        k_nx     = exp(z_nx)*k(t)^alpha + (1-delta)*k(t) - c_nx;
%        if k_nx < (1-delta)*k(t)
%            k_nx = (1-delta)*k(t);
%            c_nx = exp(z_nx)*k(t)^alpha + (1-delta)*k(t) - k_nx;
%        end
        rhsee_dr(t) = rhsee_dr(t) + wgt_nodes(n) * beta*( c_nx^(-sigma)*( exp(z_nx)*alpha*k(t)^(alpha-1) + 1-delta ) );

    end % end of integration

end % end of one period

% Standard PEA

% Training
seq_state = [k(t_drop:t_max-1);z(t_drop+1:t_max)];
seq_rhsee = rhsee_dr(t_drop+1:t_max);
pea_net   = train(net,seq_state,seq_rhsee);
new_rhsee = pea_net([k(t_drop:t_max-1);z(t_drop+1:t_max)]);

% Check convergence of policy variables
con_lev = max( abs( ( new_rhsee - rhsee(t_drop+1:end) )./rhsee(t_drop+1:end) ) );
rsq     = 1 - sum( (rhsee(t_drop+1:t_max) - new_rhsee ).^2 ) ./ sum( (rhsee(t_drop+1:t_max) - mean(rhsee(t_drop+1:t_max))).^2 );

% display results
fprintf('#################### iteration:')
disp(iter)
fprintf('Convergence:')
disp(con_lev')
fprintf('R-square:   ')
disp(rsq')
    
if max(con_lev(:)) < tol && min(rsq)>0.99
    break
end

end % end of iterations

%% Simulate to obtain the sequences

irr_rhsee = zeros(1,t_max);
irr_c     = zeros(1,t_max);
irr_k     = zeros(1,t_max);
irr_bind  = zeros(1,t_max);
irr_k(1)  = sim_k(1);

% simulation
for t=2:t_max

    irr_rhsee(t) = pea_net([irr_k(t-1);z(t)]);
    irr_c(t)     = irr_rhsee(t)^(-1/sigma);
    irr_k(t)     = exp(z(t))*irr_k(t-1)^alpha + (1-delta)*irr_k(t-1) - irr_c(t);
    if irr_k(t) < (1-delta)*irr_k(t-1)
        irr_k(t) = (1-delta)*irr_k(t-1);
        irr_c(t) = exp(z(t))*irr_k(t-1)^alpha + (1-delta)*irr_k(t-1) - irr_k(t);
        irr_bind(t) = 1;
    end

end

%% Plot

t_start = 500;
t_plot  = 50;

figure
subplot(1,2,1)
plot(t_start:t_start+t_plot,sim_c(t_start+1:t_start+1+t_plot),'b','LineWidth',3);
hold on
plot(t_start:t_start+t_plot,irr_c(t_start+1:t_start+1+t_plot),'r--','LineWidth',3);
title('c')

subplot(1,2,2)
plot(t_start:t_start+t_plot,sim_k(t_start+1:t_start+1+t_plot),'b','LineWidth',3);
hold on
plot(t_start:t_start+t_plot,irr_k(t_start+1:t_start+1+t_plot),'r--','LineWidth',3);
title('k')

return