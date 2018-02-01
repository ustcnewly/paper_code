function w_arr = compute_WSDG_w(combo_h_ftr, bag_list, cate_list, domain_list, alpha, param)

ftr_dim = size(combo_h_ftr{1},2);
w_dim = ftr_dim*param.cate_num*param.domain_num;
w_arr = zeros(w_dim, param.h_iter);

for iter = 1:param.h_iter
    w = zeros(1,w_dim);
    combo_ftr = combo_h_ftr{iter};
    for ci = 1:param.cate_num
        for dmi = 1:param.domain_num
            % top part
            sub_cate_idx = find(cate_list==ci);
            sub_domain_idx = find(domain_list==-1);
            sub_idx = intersect(sub_cate_idx, sub_domain_idx);
            bag_idx = bag_list(sub_idx);

            w_seg = alpha(sub_idx)'*combo_ftr((bag_idx-1)*param.domain_num+dmi,:);
            % bottom part
            sub_cate_idx = find(cate_list==ci);
            sub_domain_idx = find(domain_list==dmi);
            sub_idx = intersect(sub_cate_idx, sub_domain_idx);
            bag_idx = bag_list(sub_idx);
            w_seg = w_seg + alpha(sub_idx)'*combo_ftr(param.domain_num*param.total_bag_num+bag_idx,:);           
            
            idx = (ci-1)*param.domain_num+dmi;
            w((idx-1)*ftr_dim+1:idx*ftr_dim) = w_seg;
        end
    end

    w_arr(:,iter) = w';
end
