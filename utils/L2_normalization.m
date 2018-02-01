function X = L2_normalization(X)
% normalize each sample to be a unit vector
%
% X d-by-n matrix
d       = size(X, 1);
Xnorm   = sqrt(sum(X.^2)); 
Xnorm(Xnorm==0) = 1;
Xnorm   = 1./Xnorm;
X       = X.*repmat(Xnorm, [d, 1]);