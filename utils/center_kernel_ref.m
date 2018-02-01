function K12_c = center_kernel_ref(K12, Krr, Kr1, Kr2)


% Kr : r x r
% Kr1 : r x n1
% Kr2 : r x n2
% K12 : n1 x n2

% K12_c : n1 x n2

r  = size(Krr,1);
[n1, n2] = size(K12);
assert(r == size(Kr1,1) && r == size(Kr2,1) );
assert(n1 == size(Kr1,2) && n2 == size(Kr2,2) );


K12_c = K12 - mean(Kr1)' * ones(1,n2) - ones(n1,1) * mean(Kr2) + mean(mean(Krr));