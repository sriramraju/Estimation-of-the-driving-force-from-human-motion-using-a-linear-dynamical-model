clc;
clear all;
close all;

%%%%% LDS Work inspired from Raptis Work at UCLA Vision Lab

% data = Input your 3D accelerometer data here

block_size        = 5;      % Size of the Hankel matrices used in System Identification algorithm
system_order      = 5;     % Defines the order of the LDS
internal_iter     = 3;      % number of iterations of the reweighted l1 minimization
external_iter     = 8;     % number of alternations between system identification and input estimation
lambda            = 15;     % weight of the l1 term in the cost function
s_const           = 10^20;  % energy constraint of the impulse response of the LDS
plot_data         = false;

A = cell(1,1);
B = cell(1,1);
C = cell(1,1);
u = cell(1,1);
x_0 = cell(1,1);
cv_f = cell(1,1);

signal_length = size(data,1);
Data_mean = mean(data,1); 
Data = (data - repmat(Data_mean,signal_length,1));

index_head = 1:3;
[A,B,C,u,x_0, cv_f, H ] = large_scale_sparse_subspace_id(Data(:,cluster_index{jj}), block_size,system_order ,...
	 internal_iter,external_iter,lambda,s_const, plot_data);

subplot(211);plot(data);title('Accelerometer')
subplot(212);plot(u);title('Estimated Stimulus')