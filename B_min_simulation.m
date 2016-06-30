
function [B,  x0,M1 ] = B_min_simulation(A,C,y,u, T);
% B_MIN_SIMULATION Estimates B and initial state 
%
% function [B, x0, M1] = B_min_simulation(A,C,y,u,T);
%
% Estimates matrices B and initial state x0 by 
% minimizing the simulation error
% 
% Input::
%   A : state matrix of the LDS 
%   C : matrix of the LDS
%   y : output data organized as row vectors   
%           [ y(0) y(1) ... y(T-1)]
%
%   u : input data organized as row vectors
%           [ u(0) u(1) ... u(T-1)]
%
%   T : length of the data
%
% Output::
%  
%  B : matrix of the LDS system
%  x0: initial Condition of the state of the LDS
%  M1: Generalized Observability matrix of the LDS
%

%Copyright (C) 20010 Michalis Raptis
%
%This file is part of VLFeat, available under the terms of the
%GNU GPLv2, or (at your option) any later version.
%
% --------------------------------------------------- 
% AUTHOR Michalis Raptis
% VisionLab UCLA
% based on the System Identification code provided
%    by Alessandro Chiuso: chiuso at dei.unipd.it          
% --------------------------------------------------- 

if (nargin<4)
  disp('Wrong number of arguments')
  return
end;
if (nargin<5)
  T = length(y);
end

% Dimensions of input matrices

[m N]  = size(u);
if m>N 
  u = u'; % make it row vector
  [m N ]= size(u);
end
[p N] = size(y);
if p>N
  y = y';
  [p N] = size(y);
end



[ra  ca] = size(A);
[rc  cc] = size(C);

if ra ~= ca
  error('A must be square')  
end

if ca ~=cc
  eror('Ad and Cd must have the same number of columns')
end;

n = ca;


if p~=rc
  keyboard
  error('The rows of C must be equal with the size of the output')
end
%-----------------------------------------------------------
% Construction of the Appropriate Matrices for Minimizing
%  y(t) - y_hat(t) = y(t) -( CA^t*x(0) - sum_{i=0}^{t-1} ...
%                     C*A^(t-1-i)*Bu(i) 
%-----------------------------------------------------------

% Matrix M: will consist from 3 submatrices

M = [];

% Generalized Observability matrix ( M1*x(0))
M1 = [C; C*A];


% Matrix M3 sum_{i=0}^{t-1} C*A^(t-1-i)B*u(:,i)
%          = sum_{i=0}^{t-1} kron(u(:,i)',C)*A^(t-1-i)*vec(B)
if m>1
  MM3 = kron( u(:,1) ,C);
  for i = 1:m
    % The first p is equal with zero (y(1) is not depends from this term)
    M3(p+1:2*p,(i-1)*n+1:i*n) = MM3((i-1)*p+1:i*p,:);
  end
  
  
  M1 = zeros(p*T,n);
  M1(1:p,:) = C;
  for t = 2:T-1,
    M1(p*(t-1)+1:p*t,:)=M1(p*(t-2)+1:p*(t-1),:)*A;
    MM3 = MM3*A + kron(u(:,t),C);   
    for i = 1:m,
      M3(t*p+1:(t+1)*p,(i-1)*n+1:i*n) = MM3((i-1)*p+1:i*p,:);
    end   
  end
  t = T;
  M1(p*(t-1)+1:p*t,:)=M1(p*(t-2)+1:p*(t-1),:)*A;
else
  MM3 =  u(1,1)*C;
  
  % The first p is equal with zero (y(1) is not depends from this
  % term)
  M3(p+1:2*p,1:n) = MM3(1:p,:);
  
  M1 = zeros(p*T,n);
  M1(1:p,:) = C;
  for t = 2:T-1
    M1(p*(t-1)+1:p*t,:)=M1(p*(t-2)+1:p*(t-1),:)*A;
    MM3 = MM3*A + u(1,t)*C;
    M3(t*p+1:(t+1)*p,1:n) = MM3(1:p,:);
    
  end
  t = T;
  M1(p*(t-1)+1:p*t,:)=M1(p*(t-2)+1:p*(t-1),:)*A;
end


M = [M1  M3];

yv = [];
yv = y(:,1:T);
yv = yv(:); % Vectorization

% Solve Least Square
theta = pinv(M'*M)*M'*yv;


x0 = theta(1:n);

for t=1:m
  B(:,t)=theta(n+(t-1)*n+1:t*n+n);
end   
%-------------------------------------
%        END Algorithm
%-------------------------------------