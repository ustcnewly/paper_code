function [y,y_d, y_theta] = Max_Violated_y(alpha,X,ep)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description: With inputting the dual variable alpha in SVM, the data X and
% the balance parameter ep, this function outputs the most violated label
% vector y, the corresponding dimension y_d and the objective value y_theta.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
% alpha: n_tr*1,prediction coefficients
% X : d*n
% ep : balance constraint

% Output:
% y: the optimal label vector
% y_d: the corresonding dimension with the optimal label vector
% y_theta: the objective value

[d,n] = size(X);
y = sign(rand(n,1)-0.5);

y_theta = 0;
y_d = 0;

Y = zeros(1,n);
l_ind = find(Y ~= 0);
l = size(l_ind,2);
u_ind = find(Y == 0);
u = n - l;
eta = sum(Y);

for i = 1:d
       
    t = alpha(1:n)'.* X(i,:);
   
    [st,IX] = sort(t(u_ind));
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ub = floor((u + ep + eta)/2);
    lb = ceil((u-ep + eta)/2);
    
    if ub > u
        ub = u;
    end
    
    if lb < 0
        lb = 0;
    end
    
    j = lb;
    tmpv = st(lb+1:ub);
    pind = find(tmpv > 0);
    if size(pind,2) == 0
        j = ub;
    else
        j = pind(1) + lb - 1;
    end
    yy = ones(1,n);
    yy(l_ind) = Y(l_ind);
    yy(u_ind(IX(1:j))) = -1;
    y_the = t*yy';
    if y_the > y_theta
        y_theta = y_the;
        y_d = i;
        y = yy;
    end 
    
end
