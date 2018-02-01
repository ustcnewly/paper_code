function [K1_c, K21_c] = center_kernel(K1,K21)
%
% [K1_c,K2_c] = center_kernel(K1,K2)
% K1 : #train-by-#train
% K2 : #test-by-#train
%

l = size(K1,1);
j = ones(l,1);
K1_c = K1 - (j*j'*K1)/l - (K1*j*j')/l + ((j'*K1*j)*j*j')/(l^2);

if( nargin > 1 )
    assert(l==size(K21,2));
    tk =  (1/l)*sum(K1,1); % (1 x l)
    tl = ones(size(K21,1),1); % (n x 1)
    K21_c = K21 - ( tl * tk); % ( n x l )
    tk = (1/(size(K21,2)))*sum(K21_c,2); % ( n x 1 )   
    K21_c = K21_c - (tk * j'); % ( n x l )
end