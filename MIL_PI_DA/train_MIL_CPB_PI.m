function [model, final_obj, obj] = train_MIL_CPB_PI(yset, K, tK, param)

[n_samples, n_basekernels] = size(yset);
assert(n_samples == size(K, 1));
assert(n_samples == size(tK, 1));

degree  = 1;
d_norm   = 1;
if(exist('param', 'var')&&isfield(param, 'degree'))
    degree = param.degree;
end

if(exist('param', 'var')&&isfield(param, 'd_norm'))
    d_norm = param.d_norm;
end

MAX_ITER    = 100;
tau         = 0.001;

if(isfield(param, 'd'))
    d   = param.d;
    d   = d_norm*d/(sum(d.^degree)^(1/degree));
    coefficients                = zeros(n_basekernels, 1);
    coefficients(1:length(d))   = d;
else
    coefficients    = d_norm*ones(n_basekernels, 1)*(1/n_basekernels)^(1/degree);
end

obj             = [];
for i = 1:MAX_ITER
    [model, obj(i), wn] = return_alpha(K, tK, yset, coefficients, param);
        
    if(i == 1)
        fprintf('\tMKL-Iter#%-2d: obj = %f\n', i, obj(i));
    else
        fprintf('\tMKL-Iter#%-2d: obj = %f, abs(obj(i)-obj(i-1)) = %f\n', i, obj(i), abs(obj(i) - obj(i-1)));        
        fprintf('\t%.4f ', coefficients);
        fprintf('\n')
    end
    
    if (n_basekernels==1) || (i>1 && obj(i) >  obj(i-1)- tau*abs(obj(i)))
        break;
    end
    
    if(i == MAX_ITER)
        break;
    end    
    
    wnp     = wn.^(2/(degree+1));           
    eta     = (sum(wnp.^degree))^(1/degree); 
    coefficients    = d_norm*wnp/eta;
end

final_obj   = obj(end);
model.d     = coefficients;
end

function [model, obj, wn] = return_alpha(K, tK, yset, coefficients, param)
[n, m] = size(yset);
svm_C   = param.svm_C;
gamma   = param.gamma;

yK  = sum_y_kernels(yset, coefficients);
tmp_y = yset*coefficients;

kernel      = (K.*yK);

model       = svm_plus_train_all(tmp_y, ones(n, 1), kernel, tK, param);

alpha       = model.x(1:n);
zeta        = model.x(n+1:end);
beta        = alpha + zeta - svm_C;

q       = zeros(m, 1);
aka     = (alpha*alpha').*K;
for i = 1:m
    y   = yset(:, i);
    q(i) = y'*aka*y;
end
obj     = sum(alpha) - 0.5*(coefficients'*q) - 1/(2*gamma)*beta'*tK*beta;
wn      = coefficients.*sqrt(q);
end

function kernel = sum_y_kernels(labels, coefficients)
[n, m]   = size(labels);
for i = 1:m
    labels(:, i) = labels(:, i)*sqrt(coefficients(i));
end
kernel = labels*labels';
end

function models = svm_plus_train_all(tmp_y, labels, K, tK, param)

svm_C   = param.svm_C;
svm_C_star = param.svm_C_star;
gamma   = param.gamma;
n       = length(labels);

cates   = unique(labels);
n_class = length(cates);
if(n_class == 2)
    n_class = 1;
    cates   = [1 -1]; 
end

models  = cell(n_class, 1);
for ci = 1:n_class

    y       = (labels == cates(ci))*2 - 1;

    H       = 1/gamma*[tK tK; tK tK] + [K.*(y*y') zeros(n, n); zeros(n, 2*n)];
    H       = H + 1e-6*eye(size(H));   
    tmp     = -svm_C/gamma*sum(tK, 2);
    f       = [-1 + tmp; tmp];
    A1      = [];
    b1      = [];

    A2_lt      = [tmp_y' zeros(1, n)];
    A2_rt      = [ones(1,n/2),zeros(1,n/2),ones(1,n/2),zeros(1,n/2)]; 
    A2      = [A2_lt; A2_rt];
    b2      = [0; svm_C*n/2]; 
    lb      = zeros(2*n, 1);
    ub      = [Inf*ones(n/2,1);svm_C_star*ones(n/2,1);Inf(n,1)];    
    
   
    [x, val, flag] = quadprog(H, f, A1, b1, A2, b2, lb, ub);   

    alpha   = x(1:n);
    zeta    = x(n+1:end);
    
    use_bias = 1;
    if(use_bias)
        tilde_dec   = 1/gamma*tK*(alpha + zeta - svm_C);
        tilde_idx   = (zeta > 1e-10);
        if(all(~tilde_idx))  
            tilde_b = max(-tilde_dec);
        else
            tilde_b = mean(-tilde_dec(tilde_idx));
        end
        
        dec     = K*(alpha.*y);
        dec_2   = -dec + y.*(1 -  tilde_dec - tilde_b);
        tmp_idx =(alpha>1e-10);
        if(all(~tmp_idx))  
            lb  = max(dec_2(y>0));
            ub  = min(dec_2(y<0));
            b   = (lb + ub)/2;
        else
            b = mean(dec_2(tmp_idx));
        end
        if(isnan(b))
            error('b is NaN.\n');
        end
        rho = -b;
    end

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
end
