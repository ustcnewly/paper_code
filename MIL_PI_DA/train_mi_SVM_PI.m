function models = train_mi_SVM_PI(labels, K, tK, param)

svm_C   = param.svm_C;
gamma   = param.gamma;

cates   = unique(labels);
n_class = length(cates);
n       = length(labels);

if(n_class == 2)
    n_class = 1;
    cates   = [1 -1]; 
end

models  = cell(n_class, 1);
for ci = 1:n_class
    y       = (labels == cates(ci))*2 - 1;

    H       = 1/gamma*[tK tK; tK tK] + [K.*(y*y') zeros(n, n); zeros(n, 2*n)];
    tmp     = -svm_C/gamma*sum(tK, 2);
    f       = [-1 + tmp; tmp];
    A1      = [];
    b1      = [];
    
    A2_rt   = [ones(1,n/2),zeros(1,n/2),ones(1,n/2),zeros(1,n/2)];
    A2      = [y' zeros(1, n); A2_rt];  
    b2      = [0; svm_C*n/2];     
    lb      = zeros(2*n, 1);
    ub      = [Inf(n/2,1);ones(n/2,1);Inf(n,1)];     

    x       = quadprog(H, f, A1, b1, A2, b2, lb, ub);    

    alpha   = x(1:n);
    zeta    = x(n+1:end);

    tilde_dec   = 1/gamma*tK*(alpha + zeta - svm_C);
    tilde_idx   = (zeta > 1e-10);
    if(all(~tilde_idx)) 
        tilde_b     = max(-tilde_dec);
    else
        tilde_b     = mean(-tilde_dec(tilde_idx));
    end
    
    dec     = K*(alpha.*y);
    dec_2   = -dec + y.*(1 -  tilde_dec - tilde_b);
    tmp_idx =(alpha>1e-10);
    if(all(~tmp_idx))         
        lb  = max(dec_2(y>0));
        ub  = min(dec_2(y<0));
        b       = (lb + ub)/2;
    else
        b = mean(dec_2(tmp_idx));
    end
    if(isnan(b))
        error('b is NaN.\n');
    end
    rho = - b;

    index   = find(alpha > 1e-10);
    sv_coef = (alpha.*y); 
    models{ci}.sv_coef  = sv_coef(index);
    models{ci}.SVs      = index;    
    models{ci}.Label(1) = 1;
    models{ci}.rho      = rho;
    
    models{ci}.x        = x;
    models{ci}.y        = labels;
    models{ci}.param    = param;

end
if(n_class == 1)
    models = models{1};
end
    
