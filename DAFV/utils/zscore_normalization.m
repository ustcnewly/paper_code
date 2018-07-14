function X = zscore_normalization(X)
% normalize samples to be zero mean and unit variance
%
% X d-by-n matrix
[~, n]  = size(X);
X       = X - mean(X, 2)*ones(1, n);
var     = std(X, 0, 2); 
var(var==0) = 1;
var     = 1./var;
X       = X.*repmat(var, [1, n]);