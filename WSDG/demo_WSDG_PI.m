function demo_WSDG_PI()
    
    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('.\NewtonKKT');

    % initialize parameter
    param.cate_num = 6;
    param.usekernel = 1;
    param.sigma = 1e-3;
    
    param.bag_num = 20;
    param.bag_size = 5;
    param.total_bag_num = param.bag_num*param.cate_num;
    
    param.MAX_H_ITER = 50;
    param.MAX_IN_ITER = 10;
    
    % load training data
    path.data_home = 'YOUR_DATA_PATH';
    load(fullfile(path.data_home, 'train_combo_data.mat'), 'visual_features', 'text_features', 'train_label');
    load(fullfile(path.data_home, 'test_labels.mat'), 'test_labels');
    load(fullfile(path.data_home, 'test_features.mat'), 'test_features');                

    param.smpl_num = length(train_label);
    param.test_num = length(test_labels); 
    aug_visual_ftr = [visual_features; ones(1,param.smpl_num)];
    aug_text_ftr = [text_features; ones(1,param.smpl_num)];
    aug_test_ftr = [test_features; ones(1,param.test_num)];
    ori_vK = aug_visual_ftr'*aug_visual_ftr;
    ori_tK = aug_text_ftr'*aug_text_ftr;  
   
    param.domain_num = 2;
    param.neighbor_dim = param.domain_num*param.cate_num;

    fprintf('init beta_mat....\n');
    param.randseed = 10;
    beta_mat = gong_latent_newtonKKT(visual_features', train_label, param.domain_num, ori_vK, param);    

    beta_mat_sum = repmat(sum(beta_mat,2),1,param.domain_num);
    norm_beta_mat = beta_mat./beta_mat_sum;

    param.C1 = 1;
    param.C2 = 1;
    param.C3 = 1;
    param.C4 = 0.1;
    param.gamma = 10;
    param.rho = 0.2;

    param.st_num = param.total_bag_num*(param.domain_num*(param.cate_num-1)+1);
    param.train_cate_lbl = train_label;

    % calculate P
    beta_diff_product = zeros(param.smpl_num, param.smpl_num);
    for di1 = 1:param.domain_num-1
        for di2 = di1+1:param.domain_num
            beta_diff = beta_mat(:,di1)-beta_mat(:,di2);
            beta_diff_product = beta_diff_product + beta_diff*beta_diff';
        end
    end
    P = beta_diff_product.*ori_vK;

    bit0 = [0; 1];
    for i = 2 : param.bag_size
        half_len = size(bit0, 1);
        bit0 = [zeros(half_len, 1), bit0; ones(half_len, 1), bit0]; 
    end;
    bit0 = bit0';
    bitsum = sum(bit0, 1);
    pos_count = round(param.bag_size * param.rho);
    y_pos = bit0(:, bitsum==pos_count);

    h_arr = repmat(y_pos(:,1),param.total_bag_num,1);
    for h_iter = 1:param.MAX_H_ITER
        param.h_iter = h_iter;
        fprintf('h_iter %d\n', h_iter);
        d_arr = 1/h_iter*ones(h_iter,1);

        % given h, derive w and d similarly as Lp-MKL
        [combo_h_vkernel, combo_h_tkernel, combo_h_vftr, combo_h_tftr, bag_list, cate_list, domain_list] = generate_WSDG_PI_prior_info(norm_beta_mat, aug_visual_ftr, aug_text_ftr, h_arr, param);

        for in_iter = 1:param.MAX_IN_ITER

            [x,f] =  solve_WSDG_PI_QP_SMO_call(combo_h_vkernel, combo_h_tkernel, bag_list, cate_list, domain_list, d_arr, param);
            alpha = x(1:param.st_num);
            theta = x(param.st_num+1:end);

            w_arr = compute_WSDG_w(combo_h_vftr,bag_list, cate_list, domain_list, alpha, param);
            tw_arr = compute_WSDG_PI_tw(combo_h_tftr,bag_list, cate_list, domain_list, alpha, theta, param);
            w_arr = w_arr*diag(d_arr);
            tw_arr = tw_arr*diag(d_arr);

            % update d using line search
            w_norm1 = sqrt(sum(w_arr.^2,1)+param.gamma*sum(tw_arr.^2,1))';
            hPh = diag(h_arr'*P*h_arr);

            % calculate the objective
            obj = -1/2*sum(w_norm1.^2./d_arr) + param.C1*f'*(alpha+theta) - param.C2*d_arr'*hPh;
            fprintf('in_iter %d: obj %f \n', in_iter, obj);
            if h_iter==1
                break;
            end
            if in_iter>1
                if abs(prev_obj-obj)<1e-3*abs(prev_obj)
                    break;
                end
            end
            prev_obj = obj;                
            d_arr = line_search_d(w_norm1, 2*param.C2*hPh);
        end        

        out_obj = obj;
        if h_iter>1 && abs(prev_out_obj-out_obj)<1e-3*abs(prev_out_obj)
            break;
        end
        prev_out_obj = out_obj;

        % add the most violated h
        h = infer_violated_h(P, norm_beta_mat, alpha, ori_vK,  h_arr, param);
        h_arr = [h_arr, h]; 
    end

    w_sum = sum(w_arr,2);
    mw = reshape(w_sum, size(aug_test_ftr,1), param.neighbor_dim);
    test_decs = aug_test_ftr'*mw;
    test_acc = compute_WSDG_acc(test_decs, test_labels, param);
    fprintf('test_acc: %f\n', test_acc);
end




