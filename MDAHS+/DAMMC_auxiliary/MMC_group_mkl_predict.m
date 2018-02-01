function [dv] = MMC_group_mkl_predict(K, model)

M = size(K,3);
assert( size(model.sv_coef,2)==M );

if model.usebias 
    K = K + 1;
end

sv_coef = zeros(size(K,2), M);
sv_coef(model.SVs,:) = model.sv_coef;
n = size(K,1);
dv = zeros(n,1);
for m = 1 : M 
    dv = dv + K(:,:, m) * sv_coef(:,m);
end
dv = dv ./ model.rho;
