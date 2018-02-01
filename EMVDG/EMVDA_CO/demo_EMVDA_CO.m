clear;clc;

% addpath
addpath('.\utils');
addpath('.\tools\libsvm-3.17\matlab');
addpath('.\graph_laplacian');

ver = version('-release');
param.year  = str2num(ver(1:4)); 

[~,IP_address] = dos('ipconfig');
param.IP_address = IP_address;

% initialize parameter
param.OUT_MAX_ITER = 30;

domain_num = 3;
src_views = [1,3];
tgt_views = 2;
feat_idx = [1,2];

fprintf('loading data....\n');  
path.data_home = 'YOUR_DATA_PATH';
load(fullfile(path.data_home, 'data_combo.mat'), 'cate_lbl', 'dm_lbl', 'features');

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
param.theta = 10^(-7);

for fti = 1:param.feat_type_num
    norm_features{fti} = features{feat_idx(fti)}';
    % augment with 1
    train_ftr{fti} = [norm_features{fti}(train_index,:),ones(param.train_num,1)];
    test_ftr{fti} = [norm_features{fti}(test_index,:),ones(param.test_num,1)];    
end       
param.dim = size(train_ftr{1},2);

for lap_NN = 5:5:20
    for lap_GraphDistanceFunction = {'cosine','euclidean'}
        for lap_GraphWeights = {'heat','distance'} 
            for lap_LaplacianNormalize = 0
                for lap_LaplacianDegree = 1:3
                    lap_opt.NN = lap_NN;
                    lap_opt.GraphDistanceFunction = lap_GraphDistanceFunction{1};
                    lap_opt.GraphWeights = lap_GraphWeights{1};
                    lap_opt.LaplacianNormalize = lap_LaplacianNormalize;
                    lap_opt.LaplacianDegree = lap_LaplacianDegree;
                    lap_opt.GraphWeightParam = 0;

                    Lte_arr = cell(param.feat_type_num,1);
                    M_arr = cell(param.feat_type_num,1);
                    for fti = 1:param.feat_type_num
                        Lte_arr{fti} = laplacian(lap_opt, test_ftr{fti});
                        M_arr{fti} = test_ftr{fti}'*Lte_arr{fti}*test_ftr{fti};
                    end
                    clear Lte_arr;

                    lastwarn('');
                    L_arr = cell(param.feat_type_num,1);
                    for fti = 1:param.feat_type_num
                        L_arr{fti} = ((1+2*param.lambda1)*eye(param.dim)+2*param.theta*M_arr{fti})\eye(param.dim);
                    end
                    [msgstr, msgid] = lastwarn;
                    if strcmp(msgid,'MATLAB:nearlySingularMatrix')
                        fprintf('catch warning %s\n',msgstr);
                        lastwarn('');
                        continue;
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

                        output_decs_arr = main_co_LRESVM(pos_ftr, neg_ftr, L_arr, M_arr, test_ftr, param);
                        for fti = 1:param.feat_type_num
                            iter_combo_test_decs(:,ci) = iter_combo_test_decs(:,ci) + sum(output_decs_arr{fti},2);
                        end
                    end                
                    
                    [~,y_pred] = max(iter_combo_test_decs,[],2);
                    [~,~,acc] = calc_confusion_matrix(y_pred, test_label); 
                end
            end
        end  
    end
end






