function [model, obj, history] = MMC_cutting_plane_fast2(data, opt)
%
% Inputs
% data.
%   K 	: n x n x M kernel matrix, where f is the number of feature types
%   X  	: n x d  fdata matrix, can be []
%   y    : n x 1 label vector, the labels of the unlabeled data should be zeros
%  opt
%   .p : Lp-norm MKL
%   .balance_ratio : if > 0, will estimated from the labeled data
%   .usebias
%   .verbose : verbose or not
%
% Outputs
%  model
%  obj 	: objective value
%  history
%	.obj
%
%
%  \min_{\mu
%

if isfield(opt, 'p')
    p = opt.p;
else
    error('p not specified');
end

if isfield(opt, 'verbose')
    verbose = opt.verbose;
else
    verbose = 0;
end

if isfield(opt, 'usebias')
    usebias = opt.usebias;
else
    error(['In ' mfilename ' : usebias not specified.']);
end

if isfield(opt, 'max_iter')
    max_iter = opt.max_iter;
else
    max_iter = 15;
end

balance_ratio = opt.br;

%%%%%%%%%%%%%%%%%%%%%%%%
y = data.y;

S = size(data.K,3);
n = size(data.K,1);
idx_u = find(y==0);
idx_l = find(y~=0);
l = length(idx_l);
u = length(idx_u);


if ~isfield(data, 'X_new') || 1
    X_new = cell(S,1);
    for s = 1 : S
        if usebias
            [U,A] = svd(data.K(:,:,s)+1);
        else
            [U,A] = svd(data.K(:,:,s));
        end
        tmp = sqrt(diag(A));
        X_new{s} = U*diag(tmp)*U';
        clear U A X;
    end
    data.X_new = X_new;
end

obj = Inf;
if verbose
    fprintf('---------------------------------------------------\n');
    fprintf('Iter  | Obj.   | Diff.Obj.     | Obj.Chg.Ratio.    | Elapse.Time. \n');
end
start_time = tic;
y_set = [];
alpha = ones(n,1)/n;
for i = 1 : max_iter
    if i==1
        y_theta = zeros(S,1); yy = zeros(n,S);
        for s = 1 : S
            X_new = data.X_new{s};
            [yy(:,s),~,y_theta(s)] = find_violated_y(y, X_new, alpha, balance_ratio);
        end
        [max_y_theta, max_idx] = max(y_theta);
        yy = yy(:,max_idx);
        init_mu = [];
    else
        y_theta = zeros(S,1); yy = zeros(n,S);
        for s = 1 : S
            X_new = data.X_new{s};
            [yy(:,s),~,y_theta(s)] = find_violated_y(y, X_new, alpha, balance_ratio);
        end
        [max_y_theta, max_idx] = max(y_theta);
        yy = yy(:,max_idx);
        init_mu = [mu, 1e-3*ones(S,1)];
    end
    y_set = [y_set, yy];
    old_obj = obj;
    
    mkl_opt.usebias = usebias;
    mkl_opt.p = p;
    mkl_opt.verbose = verbose;
    mkl_opt.init_mu = init_mu;
    [alpha, mu, obj, history_mkl] = MMC_group_mkl_train(data.K, y_set, opt.cost_vec, mkl_opt);
    if verbose
        fprintf(2,'%d     | %g    | %g  | %g    | %g\n', i, obj,obj-old_obj, (obj-old_obj)/abs(old_obj), toc(start_time));
    end
    history.obj(i,1)=obj;
    if abs((obj - old_obj)/old_obj) < 1e-2 && i >= 5
        break;
    end
end
if verbose
    disp(mu);
end

model.alpha = alpha;
model.mu = mu; % M X T
model.y_set = y_set;
model.bar_y = model.y_set * model.mu'; % n X M

model.rho = calc_rho(data.K, model.alpha, model.y_set, model.mu, opt.cost_vec, usebias);
model.SVs = find(model.alpha~=0);
model.sv_coef = zeros(n, S);
for s = 1 : S   
    model.sv_coef(:,s) = model.alpha.* model.bar_y(:,s);
end
model.sv_coef =  model.sv_coef(model.SVs,:);
model.usebias = usebias;
if usebias
    model.b = model.bar_y'*model.alpha;
else
    model.b = zeros(S,1);
end

%%%%
% No need to do normalization for the normal of target classifier
% model.w_l2norm = model.sv_coef'*K(model.SVs,model.SVs)*model.sv_coef;
% model.sv_coef = model.sv_coef ./ sqrt(model.w_l2norm);
% model.rho = model.rho ./ sqrt(model.w_l2norm);
% model.b = model.b ./ sqrt(model.w_l2norm);




function rho  = calc_rho(K, alpha, y_set, mu, cost_vec, usebias)
Q = MMC_sum_kernels(K, y_set, mu, cost_vec, usebias);
grad = Q * alpha;
ind = find(alpha > eps);
rho = abs(sum(grad(ind))/length(ind));
