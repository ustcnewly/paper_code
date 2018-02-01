function models = train_mi_svm_PI_DA(labels, K, tK, Kss, Kst, param)

svm_C1   = param.svm_C1;
gamma1   = param.gamma1;
svm_C2   = param.svm_C2;
gamma2   = param.gamma2;
B        = param.B;
epsilon  = param.epsilon;

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

    H       = [K.*(y*y')+1/gamma1*tK+1/gamma2*Kss, 1/gamma1*tK,     1/gamma2*Kss;
               1/gamma1*tK,                        1/gamma1*tK,     zeros(size(K)); 
               1/gamma2*Kss,                       zeros(size(K)),  1/gamma2*Kss];
       
    H       = H + 1e-6*eye(size(H)); 

    ns      = size(Kst,1);
    nt      = size(Kst,2);
    assert(ns==n);

    f1      = -1-svm_C1/gamma1*sum(tK,2)-svm_C2/gamma2*ns/nt*sum(Kst,2);
    f2      = -svm_C1/gamma1*sum(tK,2);
    f3      = -svm_C2/gamma2*ns/nt*sum(Kst,2);
    f       = [f1; f2; f3];


    A1_1    = [-ones(1,n)/svm_C2, zeros(1,n), -ones(1,n)/svm_C2];       
    A1_2    = [ones(1,n)/svm_C2, zeros(1,n), ones(1,n)/svm_C2];
    A1_3    = [ones(1,n)/svm_C2, zeros(1,n), ones(1,n)/svm_C2];
    A1      = [A1_1; A1_2; A1_3];
    b1      = [ns*epsilon-n; n+ns*epsilon; B*n];

    A2_1    = [y',zeros(1, 2*n)];
    A2_2    = [ones(1, n/2),zeros(1,n/2),ones(1,n/2),zeros(1, n/2), zeros(1,n)];
    A2      = [A2_1; A2_2]; 
    b2      = [0; svm_C1*n/2];     

    lb      = zeros(3*n, 1);
    ub      = [Inf(n/2,1);ones(n/2,1);Inf(n,1);Inf(n,1)];     % a_neg<c_star, set c_star=1 here  

    x       = quadprog(H, f, A1, b1, A2, b2, lb, ub);
    

    alpha   = x(1:n);
    zeta    = x(n+1:n*2);
    zeta2   = x(n*2+1:n*3);

    tilde_dec   = 1/gamma1*tK*(alpha + zeta - svm_C1);
    tilde_idx   = (zeta > 1e-10);
    if(all(~tilde_idx))  
        tilde_b     = max(-tilde_dec);
    else
        tilde_b     = mean(-tilde_dec(tilde_idx));
    end
    
    tilde_dec2   = 1/gamma2*tK*(alpha+zeta2-svm_C2) + svm_C2/gamma2*(sum(Kss,2).*(alpha+zeta2-svm_C2)) - ns/nt*svm_C2/gamma2*(sum(Kst,2).*(alpha+zeta2-svm_C2));
    tilde_idx2   = (zeta2 > 1e-10);
    if(all(~tilde_idx2))  
        tilde_b2     = max(-tilde_dec2);
    else
        tilde_b2     = mean(-tilde_dec2(tilde_idx2));
    end
    
    dec     = K*(alpha.*y);
    dec_2   = -dec + y.*(1-tilde_dec-tilde_b) + y.*(1-tilde_dec2-tilde_b2);
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

