function demo_MVDG()

    % addpath
    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('.\MVDG_auxiliary');

    % initialize parameter
    param.dataset = 'ACT42';
    param.sti = 2;
    view = [1,3];
    
    param.OUT_MAX_ITER = 30;
    neighbors = 1:20;    
    
    % load data
    fprintf('loading data....\n');
    load('combo_data.mat', 'cate_lbl', 'dm_lbl', 'depth_feature', 'rgb_feature');

    dm_index = cell(4,1);
    for dmi = 1:4
        dm_index{dmi} = find(dm_lbl==dmi);
    end

    [train_index, test_index] = select_index(dm_index, param.sti);

    % L1 normalization and zscore
    rgb_feature = L1_normalization(rgb_feature);
    dep_feature = L1_normalization(depth_feature);
    rgb_feature = zscore_normalization(rgb_feature);
    dep_feature = zscore_normalization(dep_feature);

    % augment with 1
    rgb_feature = [rgb_feature', ones(size(rgb_feature,2),1)];
    dep_feature = [dep_feature', ones(size(dep_feature,2),1)];

    rgb_train_ftr = rgb_feature(train_index,:);
    rgb_test_ftr = rgb_feature(test_index,:);    
    dep_train_ftr = dep_feature(train_index,:);
    dep_test_ftr = dep_feature(test_index,:);    

    param.test_num = length(test_index);
    param.train_num = length(train_index);

    train_label = cate_lbl(train_index);
    test_label = cate_lbl(test_index);
    param.cate_num = max(cate_lbl);
    param.max_neighbor = max(neighbors);

    % algorithm parameters
    param.C1 = 0.1;
    param.C2 = 0.1;
    param.lambda11 = 10;
    param.lambda12 = 100;
    param.lambda2 = 0.1;
    param.gamma = 100;

    test_result = struct();
    test_result.test_combo_acc_arr = zeros(param.OUT_MAX_ITER, length(neighbors));
    test_result.test_rgb_acc_arr = zeros(param.OUT_MAX_ITER, length(neighbors));
    test_result.test_dep_acc_arr = zeros(param.OUT_MAX_ITER, length(neighbors));    
    test_result.test_combo_ap_arr = zeros(param.cate_num, param.OUT_MAX_ITER, length(neighbors));
    test_result.test_rgb_ap_arr = zeros(param.cate_num, param.OUT_MAX_ITER, length(neighbors));
    test_result.test_dep_ap_arr = zeros(param.cate_num, param.OUT_MAX_ITER, length(neighbors));

    rgb_test_decs_arr = zeros(param.cate_num,param.OUT_MAX_ITER,param.test_num,param.max_neighbor);
    dep_test_decs_arr = zeros(param.cate_num,param.OUT_MAX_ITER,param.test_num,param.max_neighbor);
    terminal_iter_arr = zeros(param.cate_num,1);

    % main code
    for ci = 1:param.cate_num
        param.cate = ci;
        rgb_pos_ftr = rgb_train_ftr(train_label==ci,:);
        rgb_neg_ftr = rgb_train_ftr(train_label~=ci,:);
        dep_pos_ftr = dep_train_ftr(train_label==ci,:);
        dep_neg_ftr = dep_train_ftr(train_label~=ci,:);
        t_start = tic;
        [rgb_test_decs_arr(ci,:,:,:), dep_test_decs_arr(ci,:,:,:),terminal_iter_arr(ci)] = main_co_LRESVM(rgb_pos_ftr,rgb_neg_ftr,dep_pos_ftr,dep_neg_ftr,rgb_test_ftr,dep_test_ftr,param);
        fprintf('cate %d: elapsed_time %f s\n', ci, toc(t_start));
    end                

    % test
    for iter = 1:param.OUT_MAX_ITER
        for ni = 1:length(neighbors)

            iter_combo_test_decs = zeros(param.test_num, param.cate_num);
            iter_rgb_test_decs = zeros(param.test_num, param.cate_num);
            iter_dep_test_decs = zeros(param.test_num, param.cate_num);

            for ci = 1 :param.cate_num
                idx = min(iter, terminal_iter_arr(ci));
                binary_test_label = 2*(test_label==ci)-1;

                tmp_rgb_decs = sum(reshape(rgb_test_decs_arr(ci,idx,:,1:neighbors(ni)),param.test_num,neighbors(ni)),2);
                tmp_dep_decs = sum(reshape(dep_test_decs_arr(ci,idx,:,1:neighbors(ni)),param.test_num,neighbors(ni)),2);
                % average the decision values of two views
                tmp_decs = (tmp_rgb_decs+tmp_dep_decs)/2;
                iter_combo_test_decs(:,ci) = tmp_decs;
                iter_rgb_test_decs(:,ci) = tmp_rgb_decs;
                iter_dep_test_decs(:,ci) = tmp_dep_decs;

                test_result.test_combo_ap_arr(ci,iter,ni) = calc_ap(binary_test_label, tmp_decs);
                test_result.test_rgb_ap_arr(ci,iter,ni) = calc_ap(binary_test_label, tmp_rgb_decs);
                test_result.test_dep_ap_arr(ci,iter,ni) = calc_ap(binary_test_label, tmp_dep_decs);
            end

            [~,y_pred] = max(iter_combo_test_decs,[],2);
            [~,~,acc] = calc_confusion_matrix(y_pred, test_label); 
            test_result.test_combo_acc_arr(iter,ni) = acc;
            [~,y_pred] = max(iter_rgb_test_decs,[],2);
            [~,~,acc] = calc_confusion_matrix(y_pred, test_label); 
            test_result.test_rgb_acc_arr(iter,ni) = acc;
            [~,y_pred] = max(iter_dep_test_decs,[],2);
            [~,~,acc] = calc_confusion_matrix(y_pred, test_label); 
            test_result.test_dep_acc_arr(iter,ni) = acc;
        end 
    end
    test_result.terminal_iter_arr = terminal_iter_arr;
    fprintf('acc %f\n', test_result.tet_combo_acc_arr(end,5));
end





