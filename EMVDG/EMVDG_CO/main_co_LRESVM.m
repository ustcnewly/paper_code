function output_decs_arr = main_co_LRESVM(pos_ftr, neg_ftr, test_ftr, param)

param.pos_num = size(pos_ftr{1},1);
param.neg_num = size(neg_ftr{1},1);
param.train_num = param.pos_num+param.neg_num;
param.dim = size(pos_ftr{1},2);
param.init_C = 1;

% initialize rgb_G and dep_G
init_train_label = [1;-ones(param.neg_num,1)];
kparam.kernel_type = 'linear';
para = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.init_C,0,4,1);
   
fprintf('train exemplar classifiers....\n');
G_arr = cell(param.feat_type_num,1);
for fti = 1:param.feat_type_num
    G_arr{fti} = zeros(param.dim,param.pos_num);
    for ii = 1:param.pos_num
        init_train_ftr = [pos_ftr{fti}(ii,1:end-1);neg_ftr{fti}(:,1:end-1)];
        tr_kernel = getKernel(init_train_ftr',kparam);
        fprintf('train %d/%d exemplar....\n', ii, param.pos_num);

        model = svmtrain(init_train_label, [(1:size(tr_kernel,1))' tr_kernel], para);
        ay      = full(model.sv_coef)*model.Label(1);
        idx     = full(model.SVs);
        b       = -(model.rho*model.Label(1));
        G_arr{fti}(:,ii) = [init_train_ftr(idx,:)'*ay;b];
    end
end   

output_decs_arr = cell(param.feat_type_num,1);
for fti = 1:param.feat_type_num
    output_decs_arr{fti} = zeros(param.test_num, param.max_neighbor);
end

W_arr = cell(param.feat_type_num,1);
obj2_arr = zeros(param.feat_type_num,1);
for out_iter = 1:param.OUT_MAX_ITER
    
    % fix W, update Z
    fprintf('update Z by ADM....\n');
    t_start = tic;
    [Z_arr, obj1_par] = update_Z_by_ADM(G_arr, param);
    fprintf('finish Z time: %f s\n', toc(t_start));

    % fix Z, update W,G    
    for fti = 1:param.feat_type_num
        fprintf('Update V%d W....\n',fti);
        t_start = tic;
        [W_arr{fti}, G_arr{fti}, obj2_arr(fti)] = update_W_LRESVM(Z_arr{fti}, G_arr{fti}, pos_ftr{fti}, neg_ftr{fti}, param);
        fprintf('finish RGB W time: %f s\n', toc(t_start));
    end
    
    % calculte test decision values
    for fti = 1:param.feat_type_num
        eg_decs = test_ftr{fti}*W_arr{fti};
        eg_decs = sort(eg_decs,2,'descend');
        output_decs_arr{fti} = eg_decs(:,1:param.max_neighbor);
    end
        
    % calculate training loss
    obj = obj1_par + sum(obj2_arr);
    fprintf('Iter %d: total_obj %f\n', out_iter, obj);
       
    if out_iter>1 && (abs(obj-prev_obj)/abs(obj)<1e-3 || obj>prev_obj)
        break;
    end
    
    prev_obj = obj;
end

    
    

