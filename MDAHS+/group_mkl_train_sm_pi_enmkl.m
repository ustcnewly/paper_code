function [alpha, mu_all, obj, history] = group_mkl_train_sm_pi_enmkl(K,Qt,index, y_set,cost_vec, opt )

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

max_iter = opt.in_max_iter;  
    
if isfield(opt, 'tol')
    tol = opt.tol;
else
    tol = 1e-3;
end
T = size(y_set,2);
n = size(y_set,1);
assert(n==size(K,1));
M = length(index)-1;

if isfield(opt, 'init_mu') && ~isempty(opt.init_mu)
    init_mu = opt.init_mu;
    e_vec = sqrt(sum(init_mu.^2));
    mu = init_mu./ sum(e_vec);
else
    init_mu = ones(M,T);
    e_vec = max(init_mu);
    mu = init_mu./ sum(e_vec);
end

lambda = opt.lambda;
bd = lambda*ones(M,T);
pnorm = 1;

ttt = tic;
mu_all = bd + mu;
Q = sum_kernels_pi(K,Qt, index, y_set, mu_all, cost_vec, usebias);
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
history.mu(1,:,:) = mu_all;

if verbose
    fprintf('---------------------------------------------------\n');
    fprintf('Iter  | Obj.       | Diff Beta      | DualGap   | KKT C.   | Elapse.Time.\n');
    fprintf('%d     | %g   | %g     | %g     | %g   | %g\n',...
        1, obj, Inf, Inf, Inf, toc(start_time));
end
for i = 2 : max_iter
    % update mu
    old_mu_all = mu_all;
    ttt = tic;
    mu = update_mu_pq(mu, K, index,  y_set, alpha, 2, usebias);
    bd = update_mu_lp(bd, K, index,  y_set, alpha, pnorm, usebias);
    bd = lambda*bd;

    if verbose
        fprintf('update_mu takes %g sec.\n', toc(ttt));
    end
    
    % update alpha
    ttt = tic;
    mu_all = bd + mu;
    Q = sum_kernels_pi(K, Qt, index, y_set, mu_all, cost_vec, usebias);
    if verbose
        fprintf('sum_kernels takes %g sec.\n', toc(ttt));
    end
    
    old_obj = obj;
    ttt = tic;
    [alpha, obj] = solve_svm_qp(Q, zeros(n,1));
    if verbose
        fprintf('sovle_svm_qp takes %g sec.\n', toc(ttt));
    end
    
    obj=-obj;
    if verbose
        fprintf('%d     | %g   | %g     | %g     | %g   | %g\n',...
            i, obj, max(max(abs(mu_all-old_mu_all))), Inf, Inf, toc(start_time));
    end
    history.obj(i,1)=obj;
    history.mu(i,:,:) = mu_all;
    
    if abs( (old_obj - obj)/old_obj) < tol && i >= 3
        break;
    end
end
if i==max_iter
    fprintf('Maximum number of iterations reached.\n');
end
end

function new_mu = update_mu_lp(mu, K, index, y_set, alpha, p, usebias)

T = size(y_set,2);
M = length(index)-1;
wnorm = zeros(M,T);
for t = 1 : T
    ay = alpha.*y_set(:,t);
    for m = 1 : M
        idx = [index{m}; index{M+1}];
        if usebias
            tmp = mu(m,t)*sqrt((ay(idx)'*(K(idx,idx,m)+1)*ay(idx)));
        else
            tmp = mu(m,t)*sqrt((ay(idx)'*K(idx,idx,m)*ay(idx)));
        end
        wnorm(m,t) = tmp;
    end
end

new_mu = wnorm.^(2/(p+1));

if all(new_mu(:)==0)
    return;
end

new_mu = new_mu/sum(sum(wnorm.^(2*p/(p+1)))).^(1/p);

end


function new_mu = update_mu_pq(mu, K, index, y_set, alpha, p, usebias)

T = size(y_set,2);
M = length(index)-1;
wnorm = zeros(M,T);
new_mu = zeros(M,T);
for t = 1 : T
    ay = alpha.*y_set(:,t);
    for m = 1 : M
        idx = [index{m}; index{M+1}];
        if usebias
            tmp = mu(m,t)*sqrt((ay(idx)'*(K(idx,idx,m)+1)*ay(idx)));
        else
            tmp = mu(m,t)*sqrt((ay(idx)'*K(idx,idx,m)*ay(idx)));
        end
        wnorm(m,t) = tmp;
    end
end
wnorm_sum_m = sum(wnorm.^((2*p)/(p+1)), 1).^((p-1)/(2*p));
for t = 1 : T
    for m = 1 : M
        new_mu(m,t) = wnorm(m,t).^(2/(p+1)) * wnorm_sum_m(t);
    end
end

new_mu = new_mu./ sum(wnorm_sum_m.^((p+1)/(p-1)));

end




