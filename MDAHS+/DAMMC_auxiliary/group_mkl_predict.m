function [dv] = group_mkl_predict(K, index, model)

M = length(index)-1;
assert( size(model.sv_coef,2)==M );

if model.usebias 
    K = K + 1;
end

sv_coef = zeros(size(K,2), M);
sv_coef(model.SVs,:) = model.sv_coef;
n = size(K,1);
dv = zeros(n,1);
for m = 1 : M
    idx = [index{m}; index{M+1}];  
    dv = dv + K(:,idx, m) * sv_coef(idx,m);
end
dv = dv ./ model.rho;
