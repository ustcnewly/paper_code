function [eg_decs_arr, obj_arr] = main_LR_MKL_DA(pos_ftr, neg_ftr, test_ftr, Lte, param)

train_label = [ones(param.pos_num,1);-ones(param.neg_num,1)];

kparam.kernel_type = 'linear';
train_kernel_arr = cell(param.feat_type_num,1);
test_kernel_arr = cell(param.feat_type_num,1);
virtual_kernel_arr = cell(param.feat_type_num,1);
avg_train_kernel = zeros(param.train_num, param.train_num);    
for fti = 1:param.feat_type_num
    tmp_train_ftr = [pos_ftr{fti};neg_ftr{fti}];
    train_kernel_arr{fti} = getKernel(tmp_train_ftr', kparam);        
    test_kernel_arr{fti} = getKernel(test_ftr{fti}', tmp_train_ftr', kparam);
    virtual_kernel_arr{fti} = train_kernel_arr{fti} + param.theta*test_kernel_arr{fti}'*Lte{fti}*test_kernel_arr{fti};
    avg_train_kernel = avg_train_kernel+train_kernel_arr{fti};
end   
avg_train_kernel = avg_train_kernel/param.feat_type_num;
            
%train exemplar classifiers
Q_all = (avg_train_kernel+1).*(train_label*train_label');
p = -ones(1+param.neg_num,1);
init_alpha = zeros(1+param.neg_num,1);
init_derive = p;
lb = zeros(param.neg_num+1,1);
ub = param.init_C*ones(param.neg_num+1,1);
A = zeros(1+param.neg_num, param.pos_num);
for ii = 1:param.pos_num;
    A(:,ii) = solve_QP_SMO_allQ_12(Q_all, p, lb, ub, init_derive, init_alpha, ii, param.pos_num, param.smo_max_iter, param.smo_eps_obj, param.smo_quiet_mode);                 
end

eg_decs_arr = zeros(param.test_num, param.max_neighbor);

obj_arr = [];
for out_iter = 1:param.OUT_MAX_ITER    
    % fix A,d, update B
    fprintf('out_iter %d update Z by SVD....\n',out_iter);
    [U,S,V] = svd(A);
    S = max(S-param.lambda1/param.lambda2,0);
    B = U*S*V';
    
    % fix B, update A,d
    fprintf('out_iter %d update A,d....\n',out_iter);
    d = 1/param.feat_type_num*ones(param.feat_type_num,1);
    for in_iter = 1:param.IN_MAX_ITER    
        t_start = tic;
        [A, alpha_mat, d, obj1] = update_A(virtual_kernel_arr, d, B, param);
        fprintf('\tin_iter %d obj %f time: %f s\n', in_iter, obj1, toc(t_start));
        
        if in_iter>1 && abs(prev_obj1-obj1)/abs(prev_obj1)<param.in_eps
            break;
        end
        prev_obj1 = obj1;
    end
    
    % calculte test decision values  
    avg_test_kernel = zeros(param.test_num, param.train_num);
    for fti = 1:param.feat_type_num
        avg_test_kernel = avg_test_kernel+d(fti)*test_kernel_arr{fti};
    end
    
    eg_decs = (avg_test_kernel+1)*sparse(1:param.train_num, 1:param.train_num, train_label)*alpha_mat;
    eg_decs = sort(eg_decs,2,'descend');
    eg_decs_arr = eg_decs(:,1:param.max_neighbor);
    
    % calculate the total obj
    obj = obj1 + param.lambda1*calc_nuclear_norm(B) + param.lambda2/2*norm(B,'fro')^2;
    obj_arr = [obj_arr, obj];
    fprintf('total_obj %f\n', obj);

    if out_iter>1 && abs(obj-prev_obj)/abs(obj)<param.out_eps
        break;
    end
    prev_obj = obj;
end

    
    

