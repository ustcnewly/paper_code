function [ProjectMatrix, EigenValues] = calc_pca(features)
% [ProjectMatrix EigenValues] = perform_pca()
%
% Compute the PCA projection matrix, support sparse matrix for fast
%
% Input: 
%   features : d-by-n matrix
% Output:
%   ProjectMatrix, EigenValues
%
% by LI Wen on 25 Sep, 2012
%

tic;
[dim, N]    = size(features);
mean_feat   = mean(features, 2);
features    = features - repmat(mean_feat, 1, N);
% after sub mean, it is not sparse, so we full it to parallelize
features    = full(features);
tt = toc;
fprintf('\tPCA:feature preprocssing time = %f\n', tt);

if dim <= N
    fprintf('\tDim < N, do cov decomposition\n');
    tic;
    cov = features*features';
    cov = full(cov);
    tt = toc;
    fprintf('\tPCA:cov computing time = %f\n', tt);
    tic;
    [eigVec eigVal] = eig(cov);
    tt = toc;
    fprintf('\tPCA:eig computing time = %f\n', tt);
else    
    fprintf('\tDim > N, do kernel decomposition\n');
    tic;
    cov = features'*features;
    cov = full(cov);
    tt = toc;
    fprintf('\tPCA:cov computing time = %f\n', tt);
    tic
    [eigVec eigVal] = eig(cov);
    eigVec = features*eigVec;
    eigVec = eigVec ./ repmat(sqrt(sum(eigVec.^2)), [dim, 1]);    %normalize
    tt = toc;
    fprintf('\tPCA:eig computing time = %f\n', tt);
end
[EigenValues ind]   = sort(diag(eigVal), 'descend');
ProjectMatrix       = eigVec(:, ind);
