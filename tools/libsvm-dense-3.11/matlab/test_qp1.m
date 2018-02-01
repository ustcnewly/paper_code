function test_qp1

addpath('C:\Program Files\mosek\6\toolbox\r2009b\');


%--------------------------------------
%   min \frac12 x'Hx + p'x
%   s.t 0 <= x <= 1
%       x'1 = 1
%--------------------------------------


load('heart_scale.mat');

H = heart_scale_inst*heart_scale_inst';
p = -ones(size(H,1),1);
[x1, obj1] = solve_svm_qp(H, p);


% Syntax  : [x,fval,exitflag,output,lambda]=quadprog(H,f,A,b,B,c,l,u,x0,options)
% Purpose : Solves the problem                   
%   
%             minimize     0.5*x'*H*x+f'*x    
%             subject to         A*x          <= b 
%                                B*x           = c
%                             l <= x <= u 
[x2, obj2] = quadprog(H, p, [], [], ones(1, size(H,1)), 1, zeros(size(H,1),1), ones(size(H,1),1));
fprintf('<============================>\n');
fprintf('solve_svm_qp <--> mosek quadprog\n');
fprintf('nnz:  %d<---->%d\n', numel(find(x1<1e-10)), numel(find(x2<1e-10)));
fprintf('sum(abs(x1-x2)): %g\n', sum(abs(x1-x2)));
fprintf('obj1-obj2 = %g\n', obj1-obj2);


