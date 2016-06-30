function [A,B, C,u,x_0, cv_f,H1] = ...
    large_scale_sparse_subspace_id(y,block_size,n, ...
				   MAXITER_L1, MAXITER_ALL, ...
				   lambda,s_const,plot_is_on)
% LARGE_SCALE_SPARSE_SUBSPACE_ID is the main framework 
%   
%   LARGE_SCALE_SPARSE_SUBSPACE_ID is the main framework of 
%   alternating minimization. It initializes the input driving
%   signal to a sparse signal and alternates between system
%   identification and estimation of an input signal.
%                                  
%  Inputs::
%         y    : multidimensional time series of measured outputs
%    block_size: the size of the hankel matrices
%         n    : order of the system
%                the hankel matrices should  have
%                2*n/(number of observations) number of blocks of rows
%   MAXITER_L1 : maximum number of iterations of the reweighted
%                 L1 minimization
%  MAXITER_ALL : maximum number of iterations of the alternating
%                minimization 
%  lambda      : sparsity penalty
%  s_const     : bound of the l2 norm of the LDS delta response
%  plot_is_on  : (true or false) show figures 
%  
%  Outputs::
%                
%   A, B, C    : matrices of deterministic state space LDS
%         u    : sparse estimated input signal
%         x_0  : Initial Condition
%         cv_f : Values of Cost Function
% 
%  ------------------------------------------------------
%   Problem Description::
%        Blind Deconvolution assuming a LDS of the form            
%            x_(k+1)  = A x_k + B u_k 
%            y_k      = C x_k      (1)
%        and that the input u_k is sparse and bounded, 
%        also C*A*B has bounded energy 
%
%      The goal of this function is to :
%       minimize || u||o
%         subject to (1), with respect u, A, B,C,D
%  

%Copyright (C) 20010 Michalis Raptis
%
%This file is part of VLFeat, available under the terms of the
%GNU GPLv2, or (at your option) any later version.
%
% --------------------------------------------------- %
%  AUTHOR   Michalis Raptis
%  UCLA VisionLab
% --------------------------------------------------- %

if nargin<5 
  error('Not enough Input Arguments')
end

if nargin <6
  lambda  = 1e-1;
end 

if nargin < 7
  plot_is_on = false;
end

% ----------------------------------------------------------- %
%           Adding  the required paths                        %
% ----------------------------------------------------------- %
addpath('src/subspace_fun');
addpath('src/stableLDS');
addpath('src/toeplitzmult');

% ----------------------------------------------------------- %

if size(y,1)>size(y,2)
  y =y'; % Each Row one Channel
end
T = size(y,2); % Length of the Input Signal
p = size(y,1); % Number of Output Signals


% ----------------------------------------------------------- %
%   Initialize to random Sparse input                         %
%   Minimize with respect the LDS and after refine the sparse %
%   input                                                     %
% ----------------------------------------------------------- %


rand('state',1); randn('state',1);
s = floor(0.05*T) ; % Initial guess of  number of spikes in signal
u = zeros(T,1);
q = randperm(T);
u(q(1:s),1) = (sign(randn(s,1)));
u(1,1) =1;
u(T,1) = 0;

% ----------------------------------------------------------- %
% Initialize parameters and weights                           %
% ----------------------------------------------------------- %

epsilon = 0.005; % reweighted threshold
gamma   = 1;     % initial weight of l1

wu      = gamma*ones(T,1);
ws      = gamma*ones(T-1,1);
% generate difference Matrix
past_wu = wu;
Ds = sparse(T-1, T,0);
for i = 1: T-1
	Ds(i,i) = -1; Ds(i,i+1) = 1; % 
end
sub_space_iterations= [];
kk =1;

% ----------------------------------------------------------- %
%                 Alternating Minimization                    %
% ----------------------------------------------------------- %

for j = 1: MAXITER_ALL
  % ----------------------------------------------------------- %
  % Given U : SUBSPACE IDENTIFICATION                           %
  % ----------------------------------------------------------- %
  
    
  [A,B,C,H_k,Obs,x_0,H1] = deterministic_subspace(y,u,block_size,n);
  
  % Bound the L2 norm of the LDS Response
  if norm(H1,2)> s_const
    H1 = (H1./norm(H1,2))*s_const;
  end

  % Computing the Cost Function
	
  p_outputs = size(C,1);
  
  % If H a Cell Matrix, we need to reshape the output and compute
  % the cost function for each Toeplitz part of the Convolution Kernel
  
  y_minus = reshape(y(:)-Obs*x_0,p_outputs,T);
  cv_f(kk) =  lambda*norm(past_wu.*u,1);
  H  = cell(p_outputs,1);
  Ht = cell(p_outputs,1);
  for i=1: p_outputs
    % From the first Column of the Computed Convolution Kernel
    % we create the corresponding Toeplitz matrices by extracting 
    % rows from the first column of the kernel
    
    % Toeplitz Class must be already Defined 
    %       (O(nlogn) matrix multiplication)
    A_top = toeplitzClass(T,H1(i:p_outputs:end)); % A_top
                                                  % Toeplitz Object
    H{i} =  A_top;       
    Ht{i}=  A_top';
    cv_f(kk) = cv_f(kk) + (sum((y_minus(i,:)' -A_top*u).*...
                               (y_minus(i,:)' -A_top*u)));
  end
  
  sub_space_iterations= [sub_space_iterations; kk];
  kk = kk +1;
  
  % ----------------------------------------------------------- %
  % Given A,B,C : Compute the Input Signal U
  % ----------------------------------------------------------- %
  
  for i = 1: MAXITER_L1  % Reweighted L1 Iterations
    
    rel_tol = 1e-3;% Relative Tollerance for the Dual Gap
    
    % Multi-Output System
    % H, Ht  cell arrays
    
    disp(sprintf(['\n Multidimension bounded input  reweighted l1',...
                  'heuristic iteration = %d'],i));
    %---------------------------------------
    %  Weigthed L1 optimization 
    %---------------------------------------
    [u,status]= l1_ls_multi_bounded_weighted...
        (H,Ht,T,T,reshape((y(:)-Obs*x_0),p,T)',lambda,...
         wu.^(-1),rel_tol, true);
    
    
    past_wu = wu;
    
    % Computing the Value of Cost Function
    y_minus = reshape(y(:) -Obs*x_0,p_outputs,T);
    cv_f(kk) =  lambda*norm(past_wu.*u,1);
    v_l1_term  = cv_f(kk);
    for ii=1: p
      cv_f(kk) = cv_f(kk) + sum((y_minus(ii,:)' -H{ii}*u).* ...
                                            (y_minus(ii,:)' -H{ii}*u));
    end
    
    kk = kk +1;
    % update weights
  
    wu = 1./ (epsilon + abs(u));
  
    
    % number of nonzero elements in the solution 
    nnz = length( find( abs( u) > epsilon)) ; 
    fprintf(1,['\nFound a feasible u in R^%d that has %d nonzeros '],T , nnz);
    fprintf(1,['\nThe fitting term is equal %d',...
               '\n the l1 norm is equal : %d \n '],...
            cv_f(kk-1)-v_l1_term, v_l1_term);
    
  end % loop of Reweighted L1


  %-----------------------------------------------
  % Plotting The Initial Signals- Reconstruction -
  % Input Signal - Cost Function
  %-----------------------------------------------
  if plot_is_on
    figure(1)
    clf;
    subplot(2,1,1)
    color_s={'r','b','g','m','y','c'} ;
    hold on
    plot(u)
    axis tight
    string = [ 'Input to the LDS - Sparsity Penalty :' , num2str(lambda)];
    title(string)
    xlabel('Time')
    
    hold off
    subplot(2,1,2)

    plot(y','k')
    hold on;
    observ = Obs*x_0 ;
    
    if p_outputs>0
      for jj=1:p_outputs
        hold on
        plot(observ(jj:p_outputs:end)+H{jj}*u,color_s{mod(jj,6)+1})
      end
    else
      plot(observ+H*u,color_s{1})
    end
    hold off
    axis tight
    title('Synthesized Signals (Colored Lines) - Real Data (Black Lines)')
    xlabel('Time')
    ylabel('Joint Angles-Degrees')
    
    figure(3)
    
    plot(cv_f)
    hold on
    plot(sub_space_iterations,cv_f(sub_space_iterations),'*r')
    drawnow;
    title('Cost Function - Simulation Error')
    xlabel('Number of Iterations') 
    hold off;
    pause(1);
  end
  
  % -----------------------------------%
  % Stopping Criteria                  %
  % -----------------------------------%
  if length(cv_f)>2
    if  (  abs(cv_f(end) - cv_f(end-1))) < 1e-5
      return
    end
  end
  
end % loop of Alternating Minimization


%---------------------------------------------------
%         END of Algorithm
%---------------------------------------------------