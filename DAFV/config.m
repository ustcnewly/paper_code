% the path to store data
param.home = 'data';
% the dimension after PCA
param.pca_dim = 30;
% the total number of categories
param.cate_num = 3;

% the number of Gaussian models per category
param.K_per_cate = 2;

% src domain and target domain
param.src_domain = 1;
param.tgt_domain = 2;

% parameters of Algorithm
param.max_in_iter = 100;
param.max_out_iter = 100;
param.Wc_epsilon = 1e-3;
param.init_lr = 1e-4;
param.min_lr = 1e-10;

% hyper parameters
param.lambda1 = 10^3;
param.lambda2 = 1;
param.lambda3 = 10;
param.lambda4 = 1;
param.lambda5 = 1;
param.epsilon = 10^-4;