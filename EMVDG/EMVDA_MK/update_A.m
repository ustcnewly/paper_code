function [A,alpha_mat,d,obj] = update_A(kernel_arr, d, B, param)

    tr_lbl = [ones(param.pos_num,1); -ones(param.neg_num,1)];
    
    lb = zeros(param.neg_num+1,1);
    ub = param.C*ones(param.neg_num+param.pos_num,1);

    init_alpha = zeros(param.neg_num+1,1);
    
    % calculate Q_all
    Q_all = zeros(param.train_num, param.train_num);
    for fti = 1:param.feat_type_num
        Q_all = Q_all + d(fti)*((kernel_arr{fti}+1).*(tr_lbl*tr_lbl'));
    end
    Q_all = Q_all + param.lambda2*eye(param.train_num);
    % caculate p_all
    p_all = -ones(1+param.neg_num,param.pos_num) - param.lambda2*B;
    A = zeros(param.neg_num+1, param.pos_num);
    alpha_mat = zeros(param.train_num, param.pos_num);
    for ii = 1:param.pos_num  
        %fprintf('%d/%d\n', ii, param.pos_num);
        p = p_all(:,ii);
        sel_idx = [ii,param.pos_num+1:param.pos_num+param.neg_num];
        init_derive = p;
        if param.year==2012       
            A(:,ii) = solve_QP_SMO_allQ_12(Q_all, p, lb, ub(sel_idx), init_derive, init_alpha, ii, param.pos_num, param.smo_max_iter, param.smo_eps_obj, param.smo_quiet_mode);                 
        else
            A(:,ii) = solve_QP_SMO_allQ_14(Q_all, p, lb, ub(sel_idx), init_derive, init_alpha, ii, param.pos_num, param.smo_max_iter, param.smo_eps_obj, param.smo_quiet_mode);      
        end
        alpha_mat([ii;(param.pos_num+1:param.train_num)'],ii) = A(:,ii);
    end
    
    obj = 1/2*trace(alpha_mat'*Q_all*alpha_mat) + trace(A'*p_all);
    
    w_norm_arr = zeros(param.feat_type_num,1);
    for fti = 1:param.feat_type_num
        w_norm_arr(fti) = sum(diag(alpha_mat'*((kernel_arr{fti}+1).*(tr_lbl*tr_lbl'))*alpha_mat));
    end
    d = d.*sqrt(w_norm_arr);
    d = d/sum(d);
end




