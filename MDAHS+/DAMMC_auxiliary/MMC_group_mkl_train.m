function [alpha, mu, obj, history] = MMC_group_mkl_train(K, y_set,cost_vec, opt )
%
%
%   mu = \min_{mu} J(mu)
%
%   J(mu) =  \max_{\alpha} 	-\frac12 \alpha' * (\sum_{t=1}^T mu_t K y^t y^t') * \alpha
%	            s.t.   \sum_{i=1}^n \alpha_i = 1
%				       \alpha_i >= 0
%
% Inputs
%   -K 		: n x n x M kernel matrix
%   -y_set	: n x T candidate label vectors
%   -cost_vec: n x 1 cost vector
%   -opt
%       .p : Lp-norm MKL
%       .init_mu : M x T vector as initialization of mu
%       .usebias : 1 use bias, 0 otherwise
%       .verbose
%
% Outputs
%   -alpha     	: n x 1 learned dual variables
%   -mu 	: M x T kernel coefficients
%   -obj  	: final objective value
%   -history
%       .obj : objective values computed
%       .mu  : mu values computed
%
% Author: Lin Chen
start_time = tic;
if isfield(opt, 'verbose')
    verbose = opt.verbose;
else
    verbose = 0;
end
if isfield(opt,'usebias')
    usebias = opt.usebias;
else
    error('Please specify whether to use bias or not!');
end
if isfield(opt,'max_iter')
    max_iter = opt.max_iter;
else
    max_iter = 50;
end
if isfield(opt, 'p')
    p = opt.p;
else
    error(['In ' mfilename ' : p not specified']);
end
if isfield(opt, 'tol')
    tol = opt.tol;
else
    tol = 1e-3;
end
T = size(y_set,2);
n = size(y_set,1);
assert(n==size(K,1));
M = size(K,3);

if isfield(opt, 'init_mu') && ~isempty(opt.init_mu)
    init_mu = opt.init_mu;
    e_vec = sqrt(sum(init_mu.^2));
    mu = init_mu./ sum(e_vec);
else
    init_mu = ones(M,T);
    e_vec = sqrt(sum(init_mu.^2));
    mu = init_mu./ sum(e_vec);
end
ttt = tic;
Q = MMC_sum_kernels(K, y_set, mu, cost_vec, usebias);
if verbose
    fprintf('sum_kernels takes %g sec.\n', toc(ttt));
end
ttt = tic;
[alpha, obj] = solve_svm_qp(Q, zeros(n,1));
if verbose
    fprintf('sovle_svm_qp takes %g sec.\n', toc(ttt));
end
obj=-obj;
history.obj(1,1) = obj;
history.mu(1,:,:) = mu;

if verbose
    fprintf('---------------------------------------------------\n');
    fprintf('Iter  | Obj.       | Diff Beta      | DualGap   | KKT C.   | Elapse.Time.\n');
    fprintf('%d     | %g   | %g     | %g     | %g   | %g\n',...
        1, obj, Inf, Inf, Inf, toc(start_time));
end
for i = 2 : max_iter
    % update mu
    old_mu = mu;
    ttt = tic;
    mu = update_mu(mu, K, y_set, alpha, p, usebias);
    if verbose
        fprintf('update_mu takes %g sec.\n', toc(ttt));
    end
    
    
    % update alpha
    ttt = tic;
    Q = MMC_sum_kernels(K, y_set, mu,cost_vec, usebias);
    if verbose
        fprintf('sum_kernels takes %g sec.\n', toc(ttt));
    end
    
    old_alpha = alpha; old_obj = obj;
    ttt = tic;
    [alpha, obj] = solve_svm_qp(Q, zeros(n,1));
    if verbose
        fprintf('sovle_svm_qp takes %g sec.\n', toc(ttt));
    end
    
    obj=-obj;
    if verbose
        fprintf('%d     | %g   | %g     | %g     | %g   | %g\n',...
            i, obj, max(max(abs(mu-old_mu))), Inf, Inf, toc(start_time));
    end
    history.obj(i,1)=obj;
    history.mu(i,:,:) = mu;
    
    if abs( (old_obj - obj)/old_obj) < tol && i >= 3
        break;
    end
end
if i==max_iter
    fprintf('Maximum number of iterations reached.\n');
end


function new_mu = update_mu(mu, K, y_set, alpha, p, usebias)
T = size(y_set,2);
M = size(K,3);
wnorm = zeros(M,T);
new_mu = zeros(M,T);
for t = 1 : T
    ay = alpha.*y_set(:,t);
    for m = 1 : M
        if usebias
            tmp = mu(m,t)*sqrt((ay'*(K(:,:,m)+1)*ay));
        else
            tmp = mu(m,t)*sqrt((ay'*K(:,:,m)*ay));
        end
        wnorm(m,t) = tmp;
    end
end
wnorm_sum_m = sum(wnorm.^(4/3), 1).^(1/4);
for t = 1 : T
    for m = 1 : M
        new_mu(m,t) = wnorm(m,t).^(2/3) * wnorm_sum_m(t);
    end
end

new_mu = new_mu./ sum(wnorm_sum_m.^3);


