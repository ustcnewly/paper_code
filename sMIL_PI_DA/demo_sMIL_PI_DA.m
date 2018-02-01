clear all;clc;
config;

addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
addpath('.\utils');
addpath('.\tools');

param.tdim = 2000;
param.max_bag_num   = 25;
param.bag_size      = 15;

param.svm_C1 = 1;
param.svm_C2 = 10;
param.gamma1 = 10;
param.gamma2 = 10000;
param.B = 10^10;
param.epsilon = 0;
param.kernel_type = 'gaussian';

% load training and test data
path.data_home = 'YOUR_DATA_PATH';
visual = load(fullfile(path.data_home, 'training_visual_bags.mat'), 'pos_bags', 'neg_bags');
txt = load(fullfile(path.data_home, 'training_textual_bags.mat'), 'pos_bags', 'neg_bags');  
load(fullfile(path.data_home,'test_labels.mat'),'test_labels');
load(fullfile(path.data_home, 'test_features.mat'), 'test_features');
     
% prepare the training bags
featuresA   = [];
featuresB   = [];
bag_num     = min(length(visual.pos_bags), param.max_bag_num);
pos_bags    = [];
for j = 1:bag_num
    bag     = struct();
    bag.bag_size    = param.bag_size;
    bag.features{1} = visual.pos_bags(j).features;
    bag.features{2} = txt.pos_bags(j).features(1:param.tdim,:);
    pos_bags    = [pos_bags bag]; 
    featuresA   = [featuresA bag.features{1}]; 
    featuresB   = [featuresB bag.features{2}]; 
end

neg_bags    = [];
for j = 1:bag_num
    bag     = struct();
    bag.bag_size    = param.bag_size;
    bag.features{1} = visual.neg_bags(j).features;
    bag.features{2} = txt.neg_bags(j).features(1:param.tdim,:);
    neg_bags    = [neg_bags bag]; 
    featuresA   = [featuresA bag.features{1}];
    featuresB   = [featuresB bag.features{2}];
end

% calculate kernels
[kernel,kernel_param]  = getKernel(featuresA, param);
tkernel  = getKernel(featuresB, param);
Kss  = getKernel(featuresA, kernel_param);
Kst = getKernel(featuresA, test_feature, kernel_param);
test_kernel = Kst';
n = size(tkernel,1);
mask = [ones(n/2,1);zeros(n/2,1)];
tkernel = tkernel.*(mask*mask');

% main algorithm

% model = train_sMIL_PI(kernel, tkernel, pos_bags, neg_bags, param);
model = train_sMIL_PI_DA(kernel, tkernel, Kss, Kst, pos_bags, neg_bags, param);

% test stage
decs = test_kernel*model.coef;
ap  = calc_ap_k(labels(:, cci), decs);


