function [x,status,history] = l1_ls_multi_bounded_weighted(A,varargin)
% L1_LS_MULTI_BOUNDED_WEIGHTED estimates the sparse input signal
%
% Reweighted l1-Regularized Least Squares Problem Solver Under Constraints
%
%   l1_ls_multi_bounded_weighted solves problems of the following form:
%    for i = 1: iter
%       minimize ||A*x-y||^2 + lambda*sum|W_i *x_i|,
%           subject to
%            abs(x) < = 1
%       Update: W
%    end
%
%      Equivalent formulation
%       minimize  || A* W^(-1).z  -y ||^2 + lambda * sum| z_i |
%           subject to
%               abs(z) < = W
%   where A and y are problem data and x is variable (described below).
%  ( note : the symbols are not corresponding to the ones used in
%      the code)
%
%   if A is a matrix, either sequence can be used.
%   if A is an object (with overloaded operators, Toeplitz matrix),
%       At, m, n must be provided.
%
% INPUT ::
%   A       : mxn matrix; input data. columns correspond to features.
%
%   At      : nxm matrix; transpose of A.
%   m       : number of examples (rows) of A
%   n       : number of features (column)s of A
%
%   y       : m vector; outcome.
%   lambda  : positive scalar; regularization parameter
%
%   tar_gap : relative target duality gap (default: 1e-3)
%   quiet   : boolean; suppress printing message when true (default: false)
%
%   (advanced arguments)
%       eta     : scalar; parameter for PCG termination (default: 1e-3)
%       pcgmaxi : scalar; number of maximum PCG iterations (default: 5000)
%
% OUTPUT ::
%   x       : n vector
%   status  : string; 'Solved' or 'Failed'
%
%   history : matrix of history data. columns represent (truncated) Newton
%             iterations; rows represent the following:
%            - 1st row) gap
%            - 2nd row) primal objective
%            - 3rd row) dual objective
%            - 4th row) step size
%            - 5th row) pcg iterations
%            - 6th row) pcg status flag
%
%
%
%Copyright (C) 20010 Michalis Raptis
%
%This file is part of VLFeat, available under the terms of the
%GNU GPLv2, or (at your option) any later version.
%
% --------------------------------------------------- 
% AUTHOR Michalis Raptis 
%        VisionLab UCLA
% 
% based on the work of  
%    Kwangmoo Koh <deneb1@stanford.edu>
%    Paper : A Method for Large-Scale l1-Regularized Least Squares  
%    Kwangmoo Koh, Seung-Jean Kim, and Stephen Boyd 
% --------------------------------------------------- 
 


%------------------------------------------------------------
%       INITIALIZE
%------------------------------------------------------------

% IPM PARAMETERS
MU              = 2;        % updating parameter of t
MAX_NT_ITER     = 200;      % maximum IPM (Newton) iteration

% LINE SEARCH PARAMETERS
ALPHA           = 0.01;     % minimum fraction of decrease in the objective
BETA            = 0.5;      % stepsize decrease factor
MAX_LS_ITER     = 300;      % maximum backtracking line search iteration


if ( (iscell(varargin{1}) || ~isvector(varargin{1})) && nargin >= 7)
    At = varargin{1};
    m  = varargin{2};
    n  = varargin{3};
    y  = varargin{4};
    lambda = varargin{5};
    wu  =  varargin{6};    %  Inverse of Weights
    varargin = varargin(7:end);
    
elseif (nargin >= 3)
  At = A';
  [m,n] = size(A);
  y  = varargin{1};
  lambda = varargin{2};
  varargin = varargin(3:end);
else
  if (~quiet)
    disp('Insufficient input arguments');
  end
  x = []; status = 'Failed'; history = [];
  return;
end

if size(y,1)< size(y,2)
  y =y';  % Each Column one Joint
end
[Ni p_outputs] = size(y);
if (Ni~=n)
  error('Some problem with the size of the input and the outputs')
end
% VARIABLE ARGUMENT HANDLING
t0         = min(max(1,1/lambda),2*n/1e-3);
defaults   = {1e-3,false,1e-3,5000,zeros(n,1),ones(n,1),t0};
given_args = ~cellfun('isempty',varargin);
defaults(given_args) = varargin(given_args);
[reltol,quiet,eta,pcgmaxi,x,u,t] = deal(defaults{:});


wu_inv = wu.^(-1);
cons = wu_inv; % Bounded Input  Constraints
f = [x-u;-x-u; x-cons;-x-cons];



% RESULT/HISTORY VARIABLES
pobjs = [] ; dobjs = [] ; sts = [] ; pitrs = []; pflgs = [];
pobj  = Inf; dobj  =-Inf; s   = Inf; pitr  = 0 ; pflg  = 0 ;

ntiter  = 0; lsiter  = 0; zntiter = 0; zlsiter = 0;
normg   = 0; prelres = 0; dxu =  zeros(2*n,1);

% diagxtx = diag(At*A);
diagxtx = 2*wu.^(2);

if (~quiet)
  disp(sprintf('\nSolving a problem of size (m=%d, n=%d), with lambda=%.5e',...
               m,n,lambda));
end
if (~quiet) 
  disp('-----------------------------------------------------------------------------');
end
if (~quiet)
  disp(sprintf('%5s %9s %15s %15s %13s %11s',...
               'iter','gap','primobj','dualobj','step len','pcg iters')); 
end


%------------------------------------------------------------
%               MAIN LOOP
%------------------------------------------------------------
z = zeros(n*p_outputs,1);
for ntiter = 0:MAX_NT_ITER
  At_nu = zeros(n,1); 
  
  for i =1:p_outputs % p_outputs = number of Toeplitz blocks
    z(n*(i-1)+1:n*i,1)  = A{i}*(wu.*x)-y(:,i);        % Constraint Variables
    nu(n*(i-1)+1:n*i,1) = 2*z(n*(i-1)+1:n*i,1);       % Dual Variables
    At_nu = At_nu+ (wu.*(At{i}*nu(n*(i-1)+1:n*i,1))); % Sum of
                                                      % wu.*A'*n 
  end 
		
  %------------------------------------------------------------
  %       CALCULATE DUALITY GAP
  %------------------------------------------------------------
  
  % Primal Objective
  pobj  = z'*z+lambda*norm(x,1); 
  
  % Dual Objective
  dobj  =  max(-0.25*nu'*nu- nu'*y(:)-...
               wu_inv'*(max(-lambda*(ones(n,1))-At_nu,zeros(n,1))+...
                        max(At_nu-lambda*(ones(n,1)),zeros(n,1)) ), dobj);
  
  
  
  
  gap   =  pobj - dobj;
  
  pobjs = [pobjs pobj]; dobjs = [dobjs dobj]; sts = [sts s];
  pflgs = [pflgs pflg]; pitrs = [pitrs pitr];
  
  %------------------------------------------------------------
  %   STOPPING CRITERION
  %------------------------------------------------------------
  if (~quiet)
    disp(sprintf('%4d %12.2e %15.5e %15.5e %11.1e %8d',...
                 ntiter, gap, pobj, dobj, s, pitr));
  end
  
  if (gap/abs(dobj) < reltol) 
    status  = 'Solved';
    history = [pobjs-dobjs; pobjs; dobjs; sts; pitrs; pflgs];
    if (~quiet) 
      disp('Absolute tolerance reached.');
    end
    %disp(sprintf('total pcg iters = %d\n',sum(pitrs)));
    
    x = wu.*x; % From the new variable (z = x) = Wi*x => x 
    return;
  end
  %------------------------------------------------------------
  %       UPDATE t
  %------------------------------------------------------------
  if (s >= 0.5)
    t = max(min(2*n*MU/gap, MU*t), t); % Chapter 11 CV book 
  end
  
  %------------------------------------------------------------
  %       CALCULATE NEWTON STEP
  %------------------------------------------------------------
    
  
  
  q1 = 1./(u+x);      
  q2 = 1./(u-x); 
  q3 =(wu_inv+x).^(-1); 
  q4 =(wu_inv-x).^(-1);
  d1 = (q1.^2+q2.^2)/t; 
  d2 = (q1.^2-q2.^2)/t;
  d3 = d1 + (q3.^2+ q4.^2)/t;
  


  %---------------- Calculate gradient  -------------------------%
  
  grad_z = zeros(n,1);
  for i=1:p_outputs
    grad_z = grad_z+  wu.*(At{i}*(z(n*(i-1)+1:n*i)*2));
  end
  gradphi = [grad_z-(q1-q2+q3-q4)/t; lambda*ones(n,1)-(q1+q2)/t];
  
  % ---  calculate vectors to be used in the preconditioner --%
  prb     = diagxtx+d3;
  prs     = prb.*d1-(d2.^2);
  
  % set pcg tolerance (relative)
  normg   = norm(gradphi);
  pcgtol  = min(1e-1,eta*gap/min(1,normg));
  
  if (ntiter ~= 0 && pitr == 0) pcgtol = pcgtol*0.1; end
  
  [dxu,pflg,prelres,pitr,presvec] = ...
      pcg(@AXfunc_l1_ls,-gradphi,pcgtol,pcgmaxi,@Mfunc_l1_ls,...
          [],dxu,A,At,d1,d2,d1./prs,d2./prs,prb./prs, p_outputs, d3,wu);
  
  if (pflg == 1) pitr = pcgmaxi; end
  
  dx  = dxu(1:n);
  du  = dxu(n+1:end);
  
  %------------------------------------------------------------
  %   BACKTRACKING LINE SEARCH
  %------------------------------------------------------------
  phi = z'*z+lambda*sum(u)-sum(log(-f))/t;
  s = 1.0;
  gdx = gradphi'*dxu;
  newz = zeros(n*p_outputs,1);
  for lsiter = 1:MAX_LS_ITER
    newx = x+s*dx; newu = u+s*du;
    newf = [newx-newu;-newx-newu; newx-cons;-newx-cons];
    if (max(newf) < 0)
      
      for i=1:p_outputs
        newz(n*(i-1)+1:n*i,1)   =  A{i}*(wu.*newx)-y(:,i);
      end
      newphi =  newz'*newz+lambda*sum(newu)-sum(log(-newf))/t;
      
      if (newphi-phi <= ALPHA*s*gdx)
        break;
      end
    end
      s = BETA*s;
  end
  if (lsiter == MAX_LS_ITER) break; end % exit by BLS
  
  x = newx; u = newu; f = newf;
end


%------------------------------------------------------------
%       ABNORMAL TERMINATION (FALL THROUGH)
%------------------------------------------------------------
if (lsiter == MAX_LS_ITER)
  % failed in backtracking linesearch.
  if (~quiet) disp('MAX_LS_ITER exceeded in BLS'); end
  status = 'Failed';
elseif (ntiter == MAX_NT_ITER)
  % fail to find the solution within MAX_NT_ITER
  if (~quiet) disp('MAX_NT_ITER exceeded.'); end
  status = 'Failed';
end
history = [pobjs-dobjs; pobjs; dobjs; sts; pitrs; pflgs];

x = wu.*x;
return;

%------------------------------------------------------------
%       COMPUTE AX (PCG)
%------------------------------------------------------------
function [y] = AXfunc_l1_ls(x,A,At,d1,d2,p1,p2,p3,p_outputs, d3,wu)
%
% y = hessphi*[x1;x2],
%
% where hessphi = [W*A'*A*W*2+D3 , D2;
%                  D2        , D1];

n  = length(x)/2;
x1 = x(1:n);
x2 = x(n+1:end);
dd = zeros(n,1);
for i=1:p_outputs
	 dd = dd +(wu.*(At{i}*(A{i}*(wu.*x1))*2));
end
y  = [dd+d3.*x1+d2.*x2; d2.*x1+d1.*x2];

%------------------------------------------------------------
%       COMPUTE P^{-1}X (PCG)
%------------------------------------------------------------
function [y] = Mfunc_l1_ls(x,A,At,d1,d2,p1,p2,p3, p_outputs,d3,wu)
%
% y = P^{-1}*x,
%

n  = length(x)/2;
x1 = x(1:n);
x2 = x(n+1:end);

y = [ p1.*x1-p2.*x2;...
     -p2.*x1+p3.*x2];


%------------------------------------------------------------
%                                END of ALGORITHM
%------------------------------------------------------------