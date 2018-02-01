function [y, flag] = infer_violated_h(P, beta_mat, alpha, tr_gK, y_arr, param)
% first cacluate the middle kernel 
% then enumerate bags to find the most violated h
        
    % reshape alpha
    alpha_ins = kron(alpha(1:param.total_bag_num), 1/param.bag_size*ones(param.bag_size,1));
    
    % construct middle matrix
    neighbor = -ones(param.smpl_num, param.neighbor_dim);
    for ii = 1:param.smpl_num
        ci = param.train_cate_lbl(ii);
        neighbor(ii,(ci-1)*param.domain_num+1:ci*param.domain_num) = param.domain_num*(param.cate_num-1)*beta_mat(ii,:);
    end
    common_neighbor = neighbor*neighbor';
    Q = common_neighbor.*tr_gK.*(alpha_ins*alpha_ins');
    
    aka = Q+2*param.C2*P;
    
    val_arr = zeros(size(y_arr,2),1);
    for yi = 1:size(y_arr,2)
        val_arr(yi) = y_arr(:,yi)'*aka*y_arr(:,yi);
    end
    max_val = max(val_arr);
    
    % generate qualified label set
    bit0 = [0; 1];
    for i = 2 : param.bag_size
        half_len = size(bit0, 1);
        bit0 = [zeros(half_len, 1), bit0; ones(half_len, 1), bit0]; %#ok<AGROW>
    end;
    bit0 = bit0';
    bitsum = sum(bit0, 1);
    pos_count = round(param.bag_size * param.rho);
    y_pos = bit0(:, bitsum==pos_count);
    y_count = size(y_pos,2);

    max_obj_val  = -1;
    max_obj_val2 = 0;
    inner_iter = 0;
    y = zeros(param.smpl_num,1);
    for bi = 1:param.total_bag_num
        y((bi-1)*param.bag_size+1:bi*param.bag_size) = [ones(pos_count,1);zeros(param.bag_size-pos_count,1)];
    end

    while (inner_iter <= 1 || (max_obj_val2 - max_obj_val) > 1e-5 * max_obj_val)

        max_obj_val = max_obj_val2;
        start = 1;
        for k = 1 : param.total_bag_num
            ori_sub_y = repmat(y(start:start+param.bag_size-1),1,y_count);
            delta_sub_y = y_pos-ori_sub_y;

            sub_aka = aka(:,start:start+param.bag_size-1);
            delta_val = y'*sub_aka*delta_sub_y;

            sub_sub_aka = sub_aka(start:start+param.bag_size-1,:);
            delta_val = 2*delta_val+sum((sub_sub_aka*delta_sub_y).*delta_sub_y,1);  % matrix(1,m), m is #Ys

            [max_delta_val,idx] = max(delta_val);

            y(start : start + param.bag_size - 1) = y_pos(:, idx);
            max_obj_val2 = max_obj_val2+max_delta_val; 

            start = start + param.bag_size;
        end
        inner_iter = inner_iter + 1;
    end
    
    if y'*aka*y <= max_val+eps
        flag = 0;
        return;
    end
    
    
    flag = 1;
    
