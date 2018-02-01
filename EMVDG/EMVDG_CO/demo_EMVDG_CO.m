clear;clc;

% addpath
addpath('.\utils');
addpath('.\tools\libsvm-3.17\matlab');
ver = version('-release');
param.year  = str2num(ver(1:4)); 

[~,IP_address] = dos('ipconfig');
param.IP_address = IP_address;

% initialize parameter
param.dataset = 'MSR';
param.OUT_MAX_ITER = 30;

src_views = [1,3];
tgt_views = 2;
feat_idx = [1,2];

fprintf('loading data....\n');
path.data_home = 'YOUR_DATA_PATH';
load(fullfile(path.data_home,'data_combo.mat'), 'cate_lbl', 'dm_lbl', 'features');

train_index = ismember(dm_lbl, src_views);
test_index = ismember(dm_lbl, tgt_views);
param.test_num = sum(test_index);
param.train_num = sum(train_index);

train_label = cate_lbl(train_index);
test_label = cate_lbl(test_index);
param.cate_num = max(cate_lbl);
param.max_neighbor = 5;

param.feat_type_num = length(feat_idx);
norm_features = cell(param.feat_type_num,1);
train_ftr = cell(param.feat_type_num,1);
test_ftr = cell(param.feat_type_num,1);

param.C1 = 10^(-1);
param.C2 = param.C1;
param.lambda1 = 10^(1);
param.lambda2 = 10^(-1);        
param.lambda3 = 10^(0);
param.gamma = 10^(1);

for fti = 1:param.feat_type_num
    norm_features{fti} = features{feat_idx(fti)}';
    % augment with 1
    train_ftr{fti} = [norm_features{fti}(train_index,:),ones(param.train_num,1)];
    test_ftr{fti} = [norm_features{fti}(test_index,:),ones(param.test_num,1)];    
end       

% main code
iter_combo_test_decs = zeros(param.test_num, param.cate_num);   
for ci = 1:param.cate_num
    param.cate = ci;
    pos_ftr = cell(param.feat_type_num,1);
    neg_ftr = cell(param.feat_type_num,1);
    for fti = 1:param.feat_type_num
        pos_ftr{fti} = train_ftr{fti}(train_label==ci,:);
        neg_ftr{fti} = train_ftr{fti}(train_label~=ci,:);
    end

    t_start = tic;
    output_decs_arr = main_co_LRESVM(pos_ftr, neg_ftr, test_ftr, param);
    for fti = 1:param.feat_type_num
        iter_combo_test_decs(:,ci) = iter_combo_test_decs(:,ci) + sum(output_decs_arr{fti},2);     
    end
end    
[~,y_pred] = max(iter_combo_test_decs,[],2);
[~,~,acc] = calc_confusion_matrix(y_pred, test_label); 





