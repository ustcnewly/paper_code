function [model, obj, history] = DAMMC_cutting_plane_mb_sm_pi(data, opt)

p = opt.p;
verbose = opt.verbose;
usebias = opt.usebias;
max_iter = opt.out_max_iter;

opt.mmd_loss_mu0 = 0;

for nth = 1:2
    idx_s = data.domain_data_index{nth};
    db0{nth} = ones(length(idx_s),1);
end

y = data.y;

nDomains = length(data.domain_data_index);
S = nDomains - 1;
n = size(data.K,1);
idx_l = find(y~=0);
l = length(idx_l);

    balance_ratios = (1+opt.br)*sum(y==1)/sum(y==-1);% revised by xxx Jul 2014;
    
    Bnum = length(balance_ratios);
 

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
        Qinv = data.Q;
        if l > 0  
            y_set_tmp = zeros(n,S*Bnum);
            
            for s = 1 : S
                
                idx_s = data.domain_data_index{s};
                idx_t = data.domain_data_index{nDomains};              
                ys = y(idx_s);
    
                model0 = rho_svmtrain_K_Kpi(ys, data.K(idx_s,idx_s,s), Qinv(idx_s,idx_s), usebias);

                if model0.usebias
                    dv = (data.K(idx_t, idx_s(model0.SVs),s)+1) * model0.sv_coef;
                else
                    dv = data.K(idx_t, idx_s(model0.SVs),s) * model0.sv_coef;
                end
                if 1  % use bias
                    for bth = 1:Bnum
                        balance_ratio = balance_ratios(bth);
                        bsth = (bth-1)*S + s;
                        np = round(1/(1/balance_ratio + 1)*length(dv));
                        [~,sidx] = sort(dv, 'descend');
                        y_set_tmp(idx_t(sidx(1:np)),bsth) = 1;
                        y_set_tmp(idx_t(sidx(np+1:end)), bsth) = -1;
                        y_set_tmp(idx_l,bsth)=y(idx_l);
                    end
                else
                    tt = sign(dv);
                    if all(tt==-1)
                        [~,m_idx] = max(dv);
                        tt(m_idx) = 1;
                        y_set_tmp(idx_t,bsth) = tt;
                    end
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
            [~, max_idx] = max(y_theta);
            yy = yy(:,max_idx);
        end
        init_mu = [];
    else  
        yy_b = [];
        for bth = 1:Bnum
            y_theta = zeros(S,1); 
            yy = zeros(n,Bnum);
            for s = 1 : S
                %bsth = (bth-1)*S + s;
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
                [yy(:,s),~,y_theta(s)] = find_violated_y(y, X_new, alpha, balance_ratio);
                yy(idx_l,s) = y(idx_l);
            end
            [~, max_idx] = max(y_theta);
            yy = yy(:,max_idx);
            yy_b = [yy_b,yy];
        end
        yy = yy_b;
        init_mu = [mu, 1e-3*ones(S,Bnum)];
    end
    y_set = [y_set, yy];
    old_obj = obj;
    
    mkl_opt.usebias = usebias;
    mkl_opt.p = p;
    mkl_opt.verbose = verbose;
    mkl_opt.init_mu = init_mu;
    mkl_opt.db0 = db0;
    mkl_opt.in_max_iter = opt.in_max_iter;
    mkl_opt.lambda = opt.lambda;
       
    [alpha, mu, obj, ~] = group_mkl_train_sm_pi_enmkl(data.K, data.Q, data.domain_data_index, y_set, opt.cost_vec, mkl_opt);  

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


function rho  = calc_rho(K, index, alpha, y_set, mu, cost_vec, usebias)
Q = sum_kernels(K, index, y_set, mu, cost_vec, usebias);
grad = Q * alpha;
ind = find(alpha > eps);
rho = abs(sum(grad(ind))/length(ind));
