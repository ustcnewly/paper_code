function [combo_h_kernel, combo_h_ftr, bag_list, cate_list, domain_list] = generate_WSDG_prior_info(beta_mat, aug_ftr, h_arr, param)

    combo_h_kernel = [];
    combo_h_ftr = cell(param.h_iter,1);
    % generate combo_kernel
    for iter = 1:param.h_iter
        combo_ftr = [];
        for bi = 1:param.total_bag_num
            for dmi = 1:param.domain_num
                bag_h = h_arr((bi-1)*param.bag_size+1:bi*param.bag_size, iter);
                bag_beta = beta_mat((bi-1)*param.bag_size+1:bi*param.bag_size, dmi);
                bag_beta_ftr = aug_ftr(:,(bi-1)*param.bag_size+1:bi*param.bag_size)*(bag_h.*bag_beta)/param.bag_size;
                combo_ftr = [combo_ftr, bag_beta_ftr]; %#ok<AGROW>
            end
        end
        for bi = 1:param.total_bag_num
            bag_h = h_arr((bi-1)*param.bag_size+1:bi*param.bag_size, iter);
            bag_ftr = aug_ftr(:,(bi-1)*param.bag_size+1:bi*param.bag_size)*bag_h/param.bag_size;
            combo_ftr = [combo_ftr, bag_ftr]; %#ok<AGROW>
        end
        combo_h_ftr{iter} = combo_ftr';
        combo_h_kernel = [combo_h_kernel; combo_ftr'*combo_ftr]; %#ok<AGROW>
    end

    % generate bag_list, cate_list and domain_list
    bag_list = [(1:param.total_bag_num)'; kron((1:param.total_bag_num)', ones(param.domain_num*(param.cate_num-1),1))];
    cate_list = kron((1:param.cate_num)', ones(param.bag_num,1));
    domain_list = [-ones(param.total_bag_num,1); repmat((1:param.domain_num)', param.total_bag_num*(param.cate_num-1),1)];
    for ci = 1:param.cate_num
        rest_c = setdiff(1:param.cate_num,ci);
        cate_list = [cate_list; repmat(kron(rest_c',ones(param.domain_num,1)),param.bag_num,1)]; %#ok<AGROW>
    end

end

