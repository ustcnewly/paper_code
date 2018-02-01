function demo_MVDA()

    addpath('.\utils');
    addpath('.\graph_laplacian');
    addpath('.\MVDA_auxiliary');
    addpath('.\tools\libsvm-3.17\matlab');

    % initialize parameter
    param.sti = 2;
        
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
    param.dim = size(rgb_feature,2);

    % algorithm parameters
    param.C1 = 0.1;
    param.C2 = 0.1;
    param.lambda11 = 10;
    param.lambda12 = 100;
    param.lambda2 = 0.1;
    param.gamma = 100;
    param.theta = 10^-5;
    for lap_NN = 5:5:20
        for lap_GraphDistanceFunction = {'cosine'}
            for lap_GraphWeights = {'heat'} 
                for lap_LaplacianNormalize = 0
                    for lap_LaplacianDegree = 1:3
                        lap_opt.NN = lap_NN;
                        lap_opt.GraphDistanceFunction = lap_GraphDistanceFunction{1};
                        lap_opt.GraphWeights = lap_GraphWeights{1};
                        lap_opt.LaplacianNormalize = lap_LaplacianNormalize;
                        lap_opt.LaplacianDegree = lap_LaplacianDegree;
                        lap_opt.GraphWeightParam = 0;

                        rgb_Lte = laplacian(lap_opt, rgb_test_ftr);
                        dep_Lte = laplacian(lap_opt, dep_test_ftr);

                        rgb_M = rgb_test_ftr'*rgb_Lte*rgb_test_ftr;
                        dep_M = dep_test_ftr'*dep_Lte*dep_test_ftr;

                        rgb_L = inv((1+2*param.lambda12)*eye(param.dim)+2*param.theta*rgb_M);
                        dep_L = inv((1+2*param.lambda12)*eye(param.dim)+2*param.theta*dep_M);


                        % main code
                        for ci = 1:param.cate_num
                            param.cate = ci;
                            rgb_pos_ftr = rgb_train_ftr(train_label==ci,:);
                            rgb_neg_ftr = rgb_train_ftr(train_label~=ci,:);
                            dep_pos_ftr = dep_train_ftr(train_label==ci,:);
                            dep_neg_ftr = dep_train_ftr(train_label~=ci,:);
                            t_start = tic;
                            [rgb_test_decs, dep_test_decs,terminal_iter] = main_co_LRESVM_DA(rgb_pos_ftr,rgb_neg_ftr,dep_pos_ftr,dep_neg_ftr,rgb_L,dep_L,rgb_M,dep_M,rgb_test_ftr,dep_test_ftr,param);

                            test_result = struct();
                            test_result.rgb_test_decs = rgb_test_decs;
                            test_result.dep_test_decs = dep_test_decs;
                            test_result.terminal_iter = terminal_iter;

                            rgb_test_decs_all(:,:,:,ci) = test_result.rgb_test_decs;
                            dep_test_decs_all(:,:,:,ci) = test_result.dep_test_decs;
                            terminal_iter_arr(ci) = test_result.terminal_iter;  

                            fprintf('cate %d: elapsed_time %f s\n', ci, toc(t_start));
                        end   

                        combo_acc_arr = zeros(param.OUT_MAX_ITER, param.max_neighbor);       
                        for iter = 1:param.OUT_MAX_ITER
                            for ni = 1:length(neighbors)
                                rgb_decs = zeros(param.test_num, param.cate_num);
                                dep_decs = zeros(param.test_num, param.cate_num);

                                for ci = 1:param.cate_num
                                    tmp_decs = reshape(rgb_test_decs_all(min(terminal_iter_arr(ci),iter),:,1:ni,ci),param.test_num,ni);
                                    rgb_decs(:,ci) = mean(tmp_decs,2);
                                    tmp_decs = reshape(dep_test_decs_all(min(terminal_iter_arr(ci),iter),:,1:ni,ci),param.test_num,ni);
                                    dep_decs(:,ci) = mean(tmp_decs,2);
                                end
                                combo_decs = (rgb_decs+dep_decs)/2;
                                [~,y_pred] = max(combo_decs,[],2);
                                [~,~,acc] = calc_confusion_matrix(y_pred, test_label); 
                                combo_acc_arr(iter,ni) = acc;

                            end 
                        end
                        fprintf('acc %f\n', combo_acc_arr(end,5));
                    end
                end
            end
        end
    end
end



