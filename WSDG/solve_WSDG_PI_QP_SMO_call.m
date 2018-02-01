function [x,f0] = solve_WSDG_PI_QP_SMO_call(combo_h_vkernel, combo_h_tkernel, bag_list, cate_list, domain_list, d_arr, param)

    f0 = zeros(param.st_num,1);
    f0(1:param.total_bag_num) = param.rho;
    f = [f0;f0];
    
    init_alpha = [zeros(param.st_num,1);-(param.cate_num-1)*param.domain_num*param.C4*ones(param.total_bag_num,1);param.C4*ones(param.st_num-param.total_bag_num,1)];
    
    lb1 = [zeros(param.total_bag_num,1); -Inf(param.st_num - param.total_bag_num,1)];  
    lb2 = [-(param.cate_num-1)*param.domain_num*param.C4*ones(param.total_bag_num,1); -Inf(param.st_num - param.total_bag_num,1)];
    lb = [lb1;lb2];
    
    ub1 = [param.C1*ones(param.total_bag_num,1); zeros(param.st_num-param.total_bag_num,1)];              
    ub2 = [(param.C3-(param.cate_num-1)*param.domain_num*param.C4)*ones(param.total_bag_num,1); param.C4*ones(param.st_num-param.total_bag_num,1)];     
    ub = [ub1;ub2];
    
    dup_bag_list = [bag_list;bag_list];
    group_list = [bag_list;param.total_bag_num+bag_list];
     
    param.max_smo_iter = 5100;
    param.eps_top = 1e-3;
    neg_derive =  compute_WSDG_PI_derivative(combo_h_vkernel, combo_h_tkernel, dup_bag_list, cate_list, domain_list, init_alpha, f, lb, ub, d_arr, param.total_bag_num, param.domain_num, param.gamma);
    x =  solve_WSDG_PI_QP_SMO(combo_h_vkernel, combo_h_tkernel, group_list, dup_bag_list, cate_list, domain_list, init_alpha, neg_derive, lb, ub, d_arr, param.total_bag_num, param.domain_num, param.gamma, param.max_smo_iter, param.eps_top, 0);
end

