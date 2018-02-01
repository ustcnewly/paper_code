function model = svmtrain_csvc(K, y, C)
%--------------------------------------
%   min 0.5 (x.*y)'K(x.*y) + p'x
%   s.t 0 <= x <= C
%       x'y = 0
%--------------------------------------
model = svmtrain_csvc_p(K,y,-ones(size(K,1),1),C);