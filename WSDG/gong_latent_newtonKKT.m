function B = gong_latent_newtonKKT(X, Y, V, K, param)
% Input: 
%     X: #data-by-#dimension
%     Y: the class label of each data point
%     K: RBF kernel matrix, K_{ij} is the similarity between data points i and j
%     V: # domains to discover
% 
% use newtonKKT to solve non-convex QP
% Output: B, #data-by-#domains 

if param.usekernel == 0
    K = rbf_kernel(X, X, param.boqing_rbf_sig);
end

M = size(X,1);
C = length(unique(Y)); 

% prepare 1/2*x'*H*x + f'*x 
KK1 = size(M*V, M*V);

for v = 1 : V
%     disp(M);
%     disp(size(K));
    KK1((v-1)*M+1:v*M, (v-1)*M+1:v*M) = K;
end
KK2 = repmat(K,V,V);
H = KK2 - V*KK1;    clear KK1 KK2
H = (H+H')/2;
H = H+1e-6*eye(size(H));

% prepare A*x <= b
A = repmat(eye(M),1,V);
A = [-A; A];
b = [-ones(M,1)/(M); ones(M,1)/C];

% prepare Aeq*x = beq
IM = ones(1,M);
Aeq = zeros(V,V*M); %blkdiag(IM,IM,IM);
for v = 1 : V
    Aeq(v,(v-1)*M+1:v*M) = IM;
end
beq = ones(V,1);

Bcs = [];   bcs = [];
Yuniq = unique(Y);
for j = 1 : length(Yuniq)
    c = Yuniq(j);
    Bc = zeros(V,M*V);
    for v = 1 : V
        Bc((v-1)*1+1:v*1, (v-1)*M+1:v*M) = (Y==c)';
    end
    Bcs = [Bcs; Bc];
    bc = repmat(ones(1,M)*(Y==c)/M, V, 1);
    bcs = [bcs; bc];
end

Aeq = [Aeq; Bcs];
beq = [beq; bcs];

%%%%%%% Plug the above to some QP solver. Note that the problem is not convex --- we are maximizing a convex function. 
%%%%%%% Suppose you have solved x, a vector,
tmp_num = M*V;
% x    = quadprog(-H, zeros(tmp_num, 1), A, b, Aeq, beq, zeros(tmp_num, 1));
% x = solqp(H, Aeq, beq, zeros(tmp_num,1));

A_combo = [Aeq; -Aeq; A; -eye(tmp_num)];
b_combo = [beq; -beq; b; zeros(tmp_num,1)];

% A_combo = [A];
% b_combo = [b];

% x0 = 1/size(X,1)*ones(size(X,1)*V,1);
rand('seed', param.randseed);
x0 = rand(tmp_num, 1);
% save('x0.mat', 'x0', '-v7.3');

% x = quadprog(-H, zeros(tmp_num,1), A, b, [], [], [], []);

x = NewtonKKTqp(H, zeros(tmp_num,1) ,A_combo, b_combo, x0);
B = reshape(x,M,[]); 
end

function knl = rbf_kernel(ftr1, ftr2, sigma)
%input: ftr1: n1*d, ftr2: n2*d
%output: knl n1*n2
knl = L2_distance_2(ftr1', ftr2');
%div = 2*sigma*sigma;
%div = sigma*median(knl(:));
div = sigma*size(ftr1, 2);
knl = exp(-knl/div);
end