function test_qp2

addpath('C:\Program Files\mosek\6\toolbox\r2009b\');


%--------------------------------------
%   min \frac12 (x*y)'*H*(x*y) + p'x
%   s.t 0 <= x <= C
%       x'*y = 0
%--------------------------------------


load('heart_scale.mat');
C = 1;
H = heart_scale_inst*heart_scale_inst';
y = heart_scale_label;
p = -ones(size(H,1),1);
[x1, obj1] = solve_svm_qp2(H, y, p, C);

% Syntax  : [x,fval,exitflag,output,lambda]=quadprog(H,f,A,b,B,c,l,u,x0,options)
% Purpose : Solves the problem                   
%   
%             minimize     0.5*x'*H*x+f'*x    
%             subject to         A*x          <= b 
%                                B*x           = c
%                             l <= x <= u 
[x2, obj2] = quadprog(diag(y)*H*diag(y), p, [], [], y', 0, zeros(size(H,1),1), C*ones(size(H,1),1));
fprintf('<============================>\n');
fprintf('solve_svm_qp <--> mosek quadprog\n');
fprintf('nnz:  %d<---->%d\n', numel(find(x1<1e-8)), numel(find(x2<1e-8)));
fprintf('sum(abs(x1-x2)): %g\n', sum(abs(x1-x2)));
fprintf('(%g)-(%g) = %g\n', obj1,obj2,obj1-obj2);


