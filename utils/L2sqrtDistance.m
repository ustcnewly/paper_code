function D = L2sqrtDistance( X, Y )
%
% Inputs
%   X : m x d matrix
%   Y : n x d matrix
% Outputs
%   D : m x n matrix
%
%
%if( ~isa(X,'double') || ~isa(Y,'double'))
%    error( 'Inputs must be of type double'); 
%end
m = size(X,1); n = size(Y,1);  
D = repmat(sum(X.*X,2), 1 , n) + repmat(sum(Y.*Y, 2)', m, 1) - 2 * X * Y';
D = real(D);
