function alpha = solve_WSDG_QP_SMO_call(combo_h_kernel, bag_list, cate_list, domain_list, d_arr, param)

    total_st_num = length(domain_list);
    f = zeros(total_st_num,1);
    f(1:param.total_bag_num) = param.rho;
    
    % initial_alpha = 0, set neg_derive
    init_alpha = zeros(total_st_num,1);
    neg_derive = f;
    
    lb = [zeros(param.total_bag_num,1); -Inf(total_st_num - param.total_bag_num,1)];                     
    ub = [param.C1*ones(param.total_bag_num,1); zeros(total_st_num-param.total_bag_num,1)];              

    param.max_smo_iter = 1000000;
    param.eps_top = 1e-4;
    alpha =  solve_WSDG_QP_SMO(combo_h_kernel, bag_list, cate_list, domain_list, init_alpha, neg_derive, lb, ub, d_arr, param.total_bag_num, param.domain_num, param.max_smo_iter, param.eps_top, 0);

end

