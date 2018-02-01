function [obj,obj1,obj2,obj3] = calc_ADM_ori_obj(Z_arr, W_arr, param)

obj1 = 0;
obj2 = 0;
obj3 = 0;

for fti = 1:param.feat_type_num
    obj1 = obj1 + param.lambda2*norm(W_arr{fti}-W_arr{fti}*Z_arr{fti},'fro')^2;
    obj2 = obj2 + param.lambda3*calc_nuclear_norm(Z_arr{fti});
end
for fti1 = 1:param.feat_type_num-1
    for fti2 = fti1+1:param.feat_type_num        
        obj3 = obj3 + param.gamma*(norm(Z_arr{fti1}-Z_arr{fti2},'fro')^2);
    end
end
obj = obj1+obj2+obj3;
% fprintf('obj1 %f obj2 %f obj3 %f Z_diff %f gamma %f\n', obj1, obj2, obj3, norm(rgb_Z-dep_Z,'fro'), param.gamma);
