function [rgb_test_decs_arr, dep_test_decs_arr, out_iter] = main_co_LRESVM(rgb_pos_ftr,rgb_neg_ftr,dep_pos_ftr,dep_neg_ftr,rgb_test_ftr,dep_test_ftr,param)

param.pos_num = size(rgb_pos_ftr,1);
param.neg_num = size(rgb_neg_ftr,1);
param.train_num = param.pos_num+param.neg_num;
param.dim = size(rgb_pos_ftr,2);
param.init_C = 1;

% initialize rgb_G and dep_G
init_train_label = [1;-ones(param.neg_num,1)];
kparam.kernel_type = 'linear';
para = sprintf('-c %.6f -s %d -t %d -w1 %.6f -q 1',param.init_C,0,4,1);
   

fprintf('train exemplar classifiers....\n');
rgb_G = zeros(param.dim,param.pos_num);
for ii = 1:param.pos_num
    init_train_ftr = [rgb_pos_ftr(ii,1:end-1);rgb_neg_ftr(:,1:end-1)];
    tr_kernel = getKernel(init_train_ftr',kparam);
    fprintf('train RGB %d/%d exemplar....\n', ii, param.pos_num);

    model = svmtrain(init_train_label, [(1:size(tr_kernel,1))' tr_kernel], para);
    ay      = full(model.sv_coef)*model.Label(1);
    idx     = full(model.SVs);
    b       = -(model.rho*model.Label(1));
    rgb_G(:,ii) = [init_train_ftr(idx,:)'*ay;b];
end
dep_G = zeros(param.dim,param.pos_num);
for ii = 1:param.pos_num
    init_train_ftr = [dep_pos_ftr(ii,1:end-1);dep_neg_ftr(:,1:end-1)];
    tr_kernel = getKernel(init_train_ftr',kparam);  
    fprintf('train dep %d/%d exemplar....\n', ii, param.pos_num);

    model = svmtrain(init_train_label, [(1:size(tr_kernel,1))' tr_kernel], para);
    ay      = full(model.sv_coef)*model.Label(1);
    idx     = full(model.SVs);
    b       = -(model.rho*model.Label(1));
    dep_G(:,ii) = [init_train_ftr(idx,:)'*ay;b];
end


rgb_test_decs_arr = zeros(param.OUT_MAX_ITER, param.test_num, param.max_neighbor);
dep_test_decs_arr = zeros(param.OUT_MAX_ITER, param.test_num, param.max_neighbor);

for out_iter = 1:param.OUT_MAX_ITER
    
    % fix W, update Z

    fprintf('update Z by ADM....\n');
    t_start = tic;
    [rgb_Z, dep_Z, obj1_par] = update_Z_by_ADM(rgb_G, dep_G, param);
    fprintf('finish Z time: %f s\n', toc(t_start));


    % fix Z, update W,G
    
    fprintf('Update RGB W....\n');
    param.w_flag = 'rgb';
    t_start = tic;
    [rgb_W, rgb_G,obj2_rgb] = update_W_LRESVM(rgb_Z, rgb_G, rgb_pos_ftr, rgb_neg_ftr, param);
    fprintf('finish RGB W time: %f s\n', toc(t_start));
    fprintf('Update depth W....\n');
    param.w_flag = 'dep';
    t_start = tic;
    [dep_W, dep_G, obj2_dep] = update_W_LRESVM(dep_Z, dep_G, dep_pos_ftr, dep_neg_ftr, param);
    fprintf('finish dep W time: %f s\n', toc(t_start));


    % calculte test decision values
    rgb_eg_decs = rgb_test_ftr*rgb_W;
    rgb_eg_decs = sort(rgb_eg_decs,2,'descend');
    rgb_test_decs_arr(out_iter,:,:) = rgb_eg_decs(:,1:param.max_neighbor);
    
    dep_eg_decs = dep_test_ftr*dep_W;
    dep_eg_decs = sort(dep_eg_decs,2,'descend');
    dep_test_decs_arr(out_iter,:,:) = dep_eg_decs(:,1:param.max_neighbor);
    
    
    % calculate training loss
    obj = obj1_par + obj2_rgb + obj2_dep;
    
    fprintf('Iter %d: total_obj %f\n', out_iter, obj);
       
    if out_iter>1 && (abs(obj-prev_obj)/abs(obj)<1e-3 || obj>prev_obj)
        break;
    end
    
    prev_obj = obj;
end

    
    

