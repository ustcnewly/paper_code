function [model, final_obj, obj] = train_MIL_CPB_PI_DA(yset, K, tK, Kss, Kst, param)

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
    [model, obj(i), wn] = return_alpha(K, tK, Kss, Kst, yset, coefficients, param);
        
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

function [model, obj, wn] = return_alpha(K, tK, Kss, Kst, yset, coefficients, param)
[n, m] = size(yset);
svm_C1   = param.svm_C1;
svm_C2   = param.svm_C2;
gamma1   = param.gamma1;
gamma2   = param.gamma2;

yK  = sum_y_kernels(yset, coefficients);
tmp_y = yset*coefficients;

kernel      = (K.*yK);

model       = svm_plus_train_all(tmp_y, ones(n, 1), kernel, tK, Kss, Kst, param);

alpha       = model.x(1:n);
zeta1       = model.x(n+1:2*n);
beta1       = alpha + zeta1 - svm_C1;
zeta2       = model.x(2*n+1:3*n);
beta2       = alpha + zeta2 - svm_C2;

q       = zeros(m, 1);
aka     = (alpha*alpha').*K;
for i = 1:m
    y   = yset(:, i);
    q(i) = y'*aka*y;
end

ns = size(Kst,1);
nt = size(Kst,2);
obj     = sum(alpha) - 0.5*(coefficients'*q) - 0.5/gamma1*beta1'*tK*beta1 -0.5/gamma2*beta2'*Kss*beta2 - svm_C2/gamma2*beta2'*sum(Kss,2) +svm_C2/gamma2*ns/nt*beta2'*sum(Kst,2);
wn      = coefficients.*sqrt(q);
end


function kernel = sum_y_kernels(labels, coefficients)
[n, m]   = size(labels);
for i = 1:m
    labels(:, i) = labels(:, i)*sqrt(coefficients(i));
end
kernel = labels*labels';
end

function models = svm_plus_train_all(tmp_y, labels, K, tK, Kss, Kst, param)

svm_C1   = param.svm_C1;
svm_C2   = param.svm_C2;
gamma1   = param.gamma1;
gamma2   = param.gamma2;
B        = param.B;
epsilon  = param.epsilon;
n        = length(labels);

cates   = unique(labels);
n_class = length(cates);
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
    H       = H + 1e-6*eye(size(H));  % avoid non-psd

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

    A2_1    = [tmp_y',zeros(1, 2*n)];
    A2_2    = [ones(1, n/2),zeros(1,n/2),ones(1,n/2),zeros(1, n/2), zeros(1,n)];
    A2      = [A2_1; A2_2];  % ay = 0, 1'(a_pos + z_pos) = sum(svm_C(1:pos_bn))
    b2      = [0; svm_C1*n/2];     

    lb      = zeros(3*n, 1);
    ub      = [Inf(n/2,1);ones(n/2,1);Inf(n,1);Inf(n,1)];     % a_neg<c_star, set c_star=1 here  

    x       = quadprog(H, f, A1, b1, A2, b2, lb, ub);
    
    if ~isempty(x)
        alpha   = x(1:n);
    else
        alpha = zeros(n,1);
    end

    % construct model
    index   = find(alpha > 1e-10);
    sv_coef = (alpha.*y); 
    models{ci}.sv_coef  = sv_coef(index);
    models{ci}.SVs      = index;    
    models{ci}.Label(1) = 1;
    
    models{ci}.x        = x;
    models{ci}.y        = labels;
    models{ci}.param    = param;
    
end
if(n_class == 1)
    models = models{1};
end
end