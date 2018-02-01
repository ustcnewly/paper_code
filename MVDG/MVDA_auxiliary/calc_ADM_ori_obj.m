function [obj,obj1,obj2,obj3] = calc_ADM_ori_obj(rgb_Z,dep_Z,rgb_W,dep_W,param)

obj1 = param.lambda11*(norm(rgb_W-rgb_W*rgb_Z,'fro')^2+norm(dep_W-dep_W*dep_Z,'fro')^2);
obj2 = param.lambda2*(calc_nuclear_norm(rgb_Z)+calc_nuclear_norm(dep_Z));
obj3 = param.gamma*(norm(rgb_Z-dep_Z,'fro')^2);
obj = obj1+obj2+obj3;
% fprintf('obj1 %f obj2 %f obj3 %f Z_diff %f gamma %f\n', obj1, obj2, obj3, norm(rgb_Z-dep_Z,'fro'), param.gamma);
