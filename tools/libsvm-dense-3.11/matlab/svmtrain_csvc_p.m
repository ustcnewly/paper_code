function model = svmtrain_csvc_p(K, y, p, C)
assert(isequal(unique(y), [-1 1]'));
assert(size(p,1)>=size(p,2));
assert(size(p,1)==size(K,1));
assert(size(K,1)==size(K,2));
n = size(p,1);


%--------------------------------------
%   min 0.5 (x.*y)'K(x.*y) + p'x
%   s.t 0 <= x <= C
%       x'y = 0
%--------------------------------------
model = svmtrain_p(p, y, [(1:n)', K], sprintf('-s 0 -t 4 -c %g -q -e %g', C, 0.001));
if y(1)<0
    model.Label = -model.Label;
    model.rho = -model.rho;
    model.sv_coef = -model.sv_coef;
    model.nSV = flipud(model.nSV);
end