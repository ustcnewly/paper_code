clear;clc;

% addpath
addpath('..\..\utils');
addpath('..\..\tools\libsvm-3.17\matlab');
ver = version('-release');
param.year  = str2num(ver(1:4)); %#ok<ST2NM>

[~,IP_address] = dos('ipconfig');
param.IP_address = IP_address;

% initialize parameter
param.OUT_MAX_ITER = 200;
param.IN_MAX_ITER = 1000;
param.init_C = 1;
param.smo_max_iter = 100000;
param.smo_eps_obj = 1e-10;
param.smo_quiet_mode = 1;
param.out_eps = 1e-3;
param.in_eps = 1e-3;

domain_num = 3;
src_views = [1,3];
tgt_views = 2;
feat_idx = [1,2];

param.C = 0.1;
param.lambda1 = 0.01;
param.lambda2 = 100;

path.data_home = 'YOUR_DATA_PATH';
load(fullfile(path.data_home,'data_combo.mat'), 'cate_lbl', 'dm_lbl', 'features');
 
train_index = ismember(dm_lbl, src_views);
test_index = ismember(dm_lbl, tgt_views);

param.test_num = sum(test_index);
param.train_num = sum(train_index);
param.total_num = param.train_num+param.test_num;

train_label = cate_lbl(train_index);
test_label = cate_lbl(test_index);
param.test_label = test_label; 
param.cate_num = max(cate_lbl);
param.max_neighbor = 5;

param.feat_type_num = length(feat_idx);
norm_features = cell(param.feat_type_num,1);
train_ftr = cell(param.feat_type_num,1);
test_ftr = cell(param.feat_type_num,1);

for fti = 1:param.feat_type_num
    norm_features{fti} = features{feat_idx(fti)}';
    train_ftr{fti} = norm_features{fti}(train_index,:);
    test_ftr{fti} = norm_features{fti}(test_index,:);    
end

obj_arr = cell(param.cate_num,1);
pos_ftr = cell(param.feat_type_num,1);
neg_ftr = cell(param.feat_type_num,1);

% main code
iter_test_decs = zeros(param.test_num, param.cate_num);
for ci = 1:param.cate_num
    param.cate = ci;
    param.pos_num = sum(train_label==ci);
    param.neg_num = param.train_num-param.pos_num;
    for fti = 1:param.feat_type_num
        pos_ftr{fti} = train_ftr{fti}(train_label==ci,:);
        neg_ftr{fti} = train_ftr{fti}(train_label~=ci,:);
    end
    [eg_decs_arr, obj_arr{ci}] = main_LR_MKL(pos_ftr, neg_ftr, test_ftr, param);
    iter_test_decs(:,ci) = sum(eg_decs_arr,2);
end                

[~,y_pred] = max(iter_test_decs,[],2);
[~,~,acc] = calc_confusion_matrix(y_pred, test_label); 
    




