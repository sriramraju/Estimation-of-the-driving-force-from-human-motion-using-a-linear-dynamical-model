function [A,B,C,H,Obs,x0,H1] = deterministic_subspace(y,u,i,n,large_min)
%  DETERMINISTIC_SUBSPACE identifies the Linear Dynamical System
%
% 
%   [A,B,C,H,Obs,x0,H1] = deterministic_subspace(y,u,i,n,large_min);
% 
%   Inputs::
%           y: matrix of measured outputs
%           u: matrix of measured inputs
%           i: number of block rows in Hankel matrices 
%              (i * #outputs) is the max. order that can be estimated 
%              Typically: i = 2 * (max order)/(#outputs)
%           n: order of the system
%   large_min: (true or false) is it a large scale minimization
%  
%    Outputs::
%           A,B,C : deterministic state space system
%           
%                  x_{k+1) = A x_k + B u_k        
%                    y_k   = C x_k 
%
%           
%   Example::
%   
%           [A,B,C,] = det_stat(y,u,10,5, true);
%           for k=3:6
%              [A,B,C] = det_stat(y,u,10,5, true);
%           end
%           
%   Reference::
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996, Page 52
%     
%    &
%           Alessandro Chiuso , Giorgio Picci 
%           System Identification Notes 2005
%
%

%Copyright (C) 20010 Michalis Raptis
%
%This file is part of VLFeat, available under the terms of the
%GNU GPLv2, or (at your option) any later version.	
%
% --------------------------------------------------- 
%   AUTHOR Michalis Raptis
%          VisionLab UCLA
%   based on the work of :
%
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%        &
%           Alessandro Chuiso, Giorgio Picci
% --------------------------------------------------- 



% Check the arguments
if (nargin < 3);
  error('det_stat needs at least three arguments');
end
if (nargin < 4);
  n = [];
end
if (nargin<5); 
  large_min = false; 
end

% Weighting is always empty
W = [];

% Turn the data into row vectors and check
[l,ny] = size(y);
if (ny < l);
  y = y';
  [l,ny] = size(y);
end
[m,nu] = size(u);
if (nu < m);
  u = u';
  [m,nu] = size(u);
end
if (i < 0);
  error('Number of block rows should be positive');
end
if (l < 0);
  error('Need a non-empty output vector');
end
if (m < 0);
  error('Need a non-empty input vector');
end
if (nu ~= ny);
  error('Number of data points different in input and output');
end
if ((nu-2*i+1) < (2*l*i));
  error('Not enough data points');
end

% Determine the number of columns in Hankel matrices
j = nu-2*i+1;

  
% Compute the R factor

U = blkhank(u/sqrt(j),2*i,j); % Input block Hankel
Y = blkhank(y/sqrt(j),2*i,j); % Output block Hankel

R = triu(qr([U;Y]'))'; 	        % R factor
R = R(1:2*i*(m+l),1:2*i*(m+l)); % Truncate
 


%---------------------------------------------------------------------------
%
%                                  BEGIN ALGORITHM
%
%---------------------------------------------------------------------------


% Assumption of Subspace identification Algorithm :
% The input u is persistenly exciting of order 2*block_size

PerEx = rank(U*U');
clear U Y
if PerEx < 2*m*i
  fprintf('*****  VIOLATION of Persistenly Exciting Assumption ****\n' )
end


%---------------------------------------
%               STEP 1 
%---------------------------------------

mi2 = 2*m*i;
% Set up some matrices
Rf = R((2*m+l)*i+1:2*(m+l)*i,:); 	% Future outputs
Rp = [R(1:m*i,:);R(2*m*i+1:(2*m+l)*i,:)]; % Past (inputs and) outputs
Ru  = R(m*i+1:2*m*i,1:mi2); 		% Future inputs
                                        % Perpendicular Future outputs 
Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(m+l)*i)]; 
% Perpendicular Past
Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(m+l)*i)]; 

% The oblique projection:
% Computed as on page 166 Formula 6.1
% obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * (Wp/Ufp)



% Funny rank check (SVD takes too long)
% This check is needed to avoid rank deficiency warnings
if (norm(Rpp(:,(2*m+l)*i-2*l:(2*m+l)*i),'fro')) < 1e-10
  Ob  = (Rfp*pinv(Rpp')')*Rp; 	% Oblique projection
else
  Ob = (Rfp/Rpp)*Rp;
end


%---------------------------------------
%               STEP 2 
%---------------------------------------

% Compute the SVD
  [U,S,V] = svd(Ob);
  ss = diag(S);
  if ss(n) < 1e-3
    disp('Probably too big System')
  end
  clear V S WOW
  


%--------------------------------------
%               STEP 3 
%--------------------------------------

% Determine the order from the singular values

U1 = U(:,1:n); 			% Determine U1



%--------------------------------------
%               STEP 4 
%--------------------------------------

% Determine gam and gamm
gam  = U1*diag(sqrt(ss(1:n)));
gamm = gam(1:l*(i-1),:);
% And their pseudo inverses
gam_inv  = pinv(gam);
gamm_inv = pinv(gamm);

%---------------------------------------
%               STEP 5
%---------------------------------------

% Estimation of A and C
S2= gam(l+1:i*l,:);

% Learn a Stable A
A = learnCGModel(gamm',S2', false);
A =A';
C = gam(1:l,:);


clear gam gamm



%--------------------------------------
%               STEP 6
%--------------------------------------

% Estimation of B  and x(O)
% by minimizing the Simulation Error

[B, x0,Obs] = B_min_simulation(A,C,y,u,length(y));



% Create the Matrix form of the System
T = length(y);
H =[];

H1 = zeros(l*T,m);

H1(l+1:end,:)= Obs(1:(T-1)*l,:)*B;


if size(C,1) ==1 && large_min ==false
  for i = 1 : T
    if i ==1
      H(:,1:m) = H1;
    else
      H((i-1)*l+1:T*l,(i-1)*m+1: m*i) = H1(1:(T-i+1)*l,1:m);
    end
  end 
else
  H =[];
end


%------------------------------------------------------------------------
%                                  END ALGORITHM
%------------------------------------------------------------------------


