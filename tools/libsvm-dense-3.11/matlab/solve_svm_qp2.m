function [x, fval] = solve_svm_qp2(H, y, p, C)
assert(size(p,1)>=size(p,2));
assert(size(p,1)==size(H,1));
assert(size(H,1)==size(H,2));
n = size(p,1);


%--------------------------------------
%   min 0.5 (x.*y)'H(x.*y) + p'x
%   s.t 0 <= x <= C
%       x'y = 0
%--------------------------------------
model = svmtrain_p(p, y, [(1:n)', H], sprintf('-s 0 -t 4 -c %g -q -e %g', C, 0.001/n));
x = zeros(n,1);
pos = full(model.SVs);
w = model.sv_coef*y(1);
x(pos) = w;
x = x.*y;
fval = 0.5*w'*H(pos, pos)*w + p(pos)'*w;
