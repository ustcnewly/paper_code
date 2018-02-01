function model = rho_svmtrain_K_Kpi(y, K, Qt, usebias)
n = size(y,1);
assert(isequal(unique(y), [-1 1]'));
assert(size(K,1)==n);
%assert(C>0);

if usebias
    %Q = (K+1).*(y*y') + 1/C*eye(n);
    Q = K .* (y*y') + Qt;
else
    %Q = K .* (y*y') + 1/C * eye(n);
    Q = K .* (y*y') + Qt;
end
[alpha, obj] = solve_svm_qp(Q, zeros(n,1));

model.obj = -obj;
model.SVs = find(alpha > 0);
model.sv_coef = alpha(model.SVs).*y(model.SVs);
model.rho = calc_rho(Q, alpha);
model.usebias = usebias;

if usebias
    model.b = sum(model.sv_coef);
else
    model.b = 0;
end

%%%
model.w_l2norm = model.sv_coef'*K(model.SVs,model.SVs)*model.sv_coef;
model.sv_coef = model.sv_coef ./ sqrt(model.w_l2norm);
model.b = model.b ./ sqrt(model.w_l2norm);
model.rho = model.rho ./ sqrt(model.w_l2norm);

function rho  = calc_rho(Q, alpha)
grad = -Q * alpha;
ind = find(alpha > eps);
rho = abs(sum(grad(ind))/length(ind));
