function [x, fval] = solve_svm_qp(H, p)
assert(size(p,1)>=size(p,2));
assert(size(p,1)==size(H,1));
assert(size(H,1)==size(H,2));
n = size(p,1);


%--------------------------------------
%   min \frac12 x'Hx + p'x
%   s.t 0 <= x <= 1
%       x'1 = 1
%--------------------------------------
model = svmtrain_p(p, ones(n,1), [(1:n)', H], sprintf('-s 2 -t 4 -n %g -q -e %g', 1/n, 0.001/n));
x = zeros(n,1);
pos = full(model.SVs);
w = model.sv_coef;
x(pos) = w;
fval = 0.5*w'*H(pos, pos)*w + p(pos)'*w;
