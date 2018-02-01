function [model, obj, history] = DAMMC_cutting_plane_fast2_mb_sm_pi(data, opt)
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
% revised by xxxing @Jul-8-2014
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

%%%%%%%%%%%%%%%%%%%%%%%%
y = data.y;

nDomains = length(data.domain_data_index);
S = nDomains - 1;
n = size(data.K,1);
idx_u = find(y==0);
idx_l = find(y~=0);
l = length(idx_l);
u = length(idx_u);


if isfield(opt, 'br')
    %balance_ratio = opt.br;
    balance_ratio = (1+opt.br)*sum(y==1)/sum(y==-1);% revised by xxx Jul 2014;
    balance_ratios = (1+opt.br)*sum(y==1)/sum(y==-1);% revised by xxx Jul 2014;
    Bnum = length(balance_ratios);
    %balance_ratio = (1+opt.br)*sum(y==1)/length(y);
else
    balance_ratio = sum(y==1)/sum(y==-1);
end

if ~isfield(data, 'X_new') || 1
    X_new = cell(S,1);
    for s = 1 : S
        idx_s = data.domain_data_index{s};
        idx_t = data.domain_data_index{nDomains};
        idx_st = [idx_s; idx_t];
        if usebias
            [U,A] = svd(data.K(idx_st,idx_st,s)+1);
        else
            [U,A] = svd(data.K(idx_st,idx_st,s));
        end
        tmp = sqrt(diag(A));
        X_new{s} = U*diag(tmp)*U';
        clear U A;
        clear X;
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
        if l > 0
            y_set_tmp = zeros(n,S*Bnum);
            for bth = 1:Bnum
            for s = 1 : S
                bsth = (bth-1)*S + s;
                balance_ratio = balance_ratios(bth);
                idx_s = data.domain_data_index{s};
                idx_t = data.domain_data_index{nDomains};              
                Cl = opt.C(s);
                ys = y(idx_s);
                %model0 = rho_svmtrain_K(ys, data.K(idx_s,idx_s,s), Cl, usebias);
                model0 = rho_svmtrain_K_Kpi(ys, data.K(idx_s,idx_s,s), data.Q(idx_s,idx_s), usebias);
                
                if model0.usebias
                    dv = (data.K(idx_t, idx_s(model0.SVs),s)+1) * model0.sv_coef;
                else
                    dv = data.K(idx_t, idx_s(model0.SVs),s) * model0.sv_coef;
                end
                if 1
                    np = round(1/(1/balance_ratio + 1)*length(dv));
                    [~,sidx] = sort(dv, 'descend');
                    y_set_tmp(idx_t(sidx(1:np)),bsth) = 1;
                    y_set_tmp(idx_t(sidx(np+1:end)), bsth) = -1;
                else
                    tt = sign(dv);
                    if all(tt==-1)
                        [~,m_idx] = max(dv);
                        tt(m_idx) = 1;
                        y_set_tmp(idx_t,bsth) = tt;
                    end
                end
                y_set_tmp(idx_l,bsth)=y(idx_l);
            end
            end
            yy = y_set_tmp;
            yy = unique(yy', 'rows')';
        else
            y_theta = zeros(S*Bnum,1); yy = zeros(n,S*Bnum);
            for bth = 1:Bnum
            for s = 1 : S
                bsth = (bth-1)*S + s;
                balance_ratio = balance_ratios(bth);
                idx_s = data.domain_data_index{s};
                idx_t = data.domain_data_index{nDomains};
                idx_st = [idx_s; idx_t];
                if usebias
                    X_new = zeros(n, size(data.X_new{s}, 2)+1);
                    X_new(idx_st,1:end-1) = data.X_new{s};
                    X_new(idx_st, end) = 1;
                else
                    X_new = zeros(n, size(data.X_new{s}, 2));
                    X_new(idx_st,:) = data.X_new{s};
                end                
               
                [yy(:,bsth),~,y_theta(bsth)] = find_violated_y(y, X_new, alpha, balance_ratio);
                yy(idx_l,bsth) = y(idx_l);
            end
            end
            [max_y_theta, max_idx] = max(y_theta);
            yy = yy(:,max_idx);
            init_mu = [];
        end
        init_mu = [];
    else       
        y_theta = zeros(S*Bnum,1); yy = zeros(n,S*Bnum);
        for bth = 1:Bnum
        for s = 1 : S
             bsth = (bth-1)*S + s;
             balance_ratio = balance_ratios(bth);
            
            
            idx_s = data.domain_data_index{s};
            idx_t = data.domain_data_index{nDomains};
            idx_st = [idx_s; idx_t];    
            if usebias
                X_new = zeros(n, size(data.X_new{s}, 2)+1);
                X_new(idx_st,1:end-1) = data.X_new{s};
                X_new(:, end) = 1;
            else
                X_new = zeros(n, size(data.X_new{s}, 2));
                X_new(idx_st,:) = data.X_new{s};
            end
            [yy(:,bsth),~,y_theta(bsth)] = find_violated_y(y, X_new, alpha, balance_ratio);
            yy(idx_l,bsth) = y(idx_l);
        end
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
    mkl_opt.opt = opt.opt;
    if opt.opt.mkl_train_type == 0
        [alpha, mu, obj, history_mkl] = group_mkl_train(data.K, data.domain_data_index, y_set, opt.cost_vec, mkl_opt);
    elseif opt.opt.mkl_train_type == 1
        [alpha, mu, obj, history_mkl] = group_mkl_train_sm(data.K, data.domain_data_index, y_set, opt.cost_vec, mkl_opt);
    elseif opt.opt.mkl_train_type == 3
        [alpha, mu, obj, history_mkl] = group_mkl_train_sm_pi(data.K, data.Q, data.domain_data_index, y_set, opt.cost_vec, mkl_opt);    
    end
    
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

model.history = history;

model.alpha = alpha;
model.mu = mu; % M X T
model.y_set = y_set;
model.bar_y = model.y_set * model.mu'; % n X M

model.rho = calc_rho(data.K, data.domain_data_index, model.alpha, model.y_set, model.mu, opt.cost_vec, usebias);
model.SVs = find(model.alpha~=0);
model.sv_coef = zeros(n, S);
for s = 1 : S
    idx_s = data.domain_data_index{s};
    idx_t = data.domain_data_index{nDomains};
    idx_st = [idx_s; idx_t];
    model.sv_coef(idx_st,s) = model.alpha(idx_st).* model.bar_y(idx_st,s);
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




function rho  = calc_rho(K, index, alpha, y_set, mu, cost_vec, usebias)
Q = sum_kernels(K, index, y_set, mu, cost_vec, usebias);
grad = Q * alpha;
ind = find(alpha > eps);
rho = abs(sum(grad(ind))/length(ind));
