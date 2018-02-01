function [y,y_d, y_theta] = find_violated_y(Y, X,alpha, br)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Description: With inputting the dual variable alpha in SVM, the data X and
% the balance parameter ep, this function outputs the most violated label
% vector y, the corresponding dimension y_d and the objective value y_theta.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:
%  Y: n x 1 label vector
%  alpha: n_tr*1,prediction coefficients
%  X : n x d
%  br : balance ratio: pos/neg

% Output:
%  y: the optimal label vector
%  y_d: the corresonding dimension with the optimal label vector
%  y_theta: the objective value

[n d] = size(X);
y_theta = 0;
y_d = 0;

l_ind = find(Y ~= 0);
l = length(l_ind);
u_ind = find(Y == 0);
u = n - l;

np = floor(u*br/(br+1));

for i = 1:d
       
    t = alpha(1:n)'.* X(:,i)';
   
    [st,IX] = sort(t(u_ind));
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    yy = -ones(n,1);
    yy(l_ind) = Y(l_ind);
    yy(u_ind(IX(1:np))) = 1;
    y_the = abs(t*yy);
    if y_the > y_theta
        y_theta = y_the;
        y_d = i;
        y = yy;
    end 
    
    yy = -ones(n,1);
    yy(l_ind) = Y(l_ind);
    yy(u_ind(IX(end-np+1:np))) = 1;
    y_the = abs(t*yy);
    if y_the > y_theta
        y_theta = y_the;
        y_d = i;
        y = yy;
    end 
end
