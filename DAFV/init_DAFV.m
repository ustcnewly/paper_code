
function init_combo = init_DAFV(data_combo, param)

    train_idx = find(data_combo.domains==param.src_domain);
    test_idx = find(data_combo.domains==param.tgt_domain);

    train_label = data_combo.labels(:,train_idx);      
    ns = length(train_idx);
    nt = length(test_idx);
    n = ns+nt;
    L = [1/(ns^2)*ones(ns), -1/(ns*nt)*ones(ns,nt); -1/(ns*nt)*ones(nt,ns), 1/(nt^2)*ones(nt)];
    H = eye(n)-1/n*ones(n);

    % Initialize R_c, W_c, S_c
    init_combo.init_Rc_arr = cell(param.cate_num,1);
    init_combo.init_Wc_arr = cell(param.cate_num,1);
    init_combo.init_Sc_arr = cell(param.cate_num,1);
    init_combo.init_decs_arr = cell(param.cate_num,1);

    init_combo.init_Wc_vals = zeros(param.cate_num,1);        
    init_combo.init_norm21_vals = zeros(param.cate_num,1);
    init_combo.init_mmd_vals = zeros(param.cate_num,1);
    init_combo.init_RS_vals = zeros(param.cate_num,1);
    init_combo.init_diff_vals = zeros(param.cate_num,1);

    init_combo.init_decs_sum = zeros(param.cate_num,ns);


    for ci = 1:param.cate_num                                                              
        Xcs = data_combo.features{ci}(:,train_idx);
        Xct = data_combo.features{ci}(:,test_idx);
        Xc = [Xcs,Xct];
        eig_vec = calc_pca(Xc);
        t1 = tic;
        init_combo.init_Rc_arr{ci} = eig_vec(:,1:param.pca_dim)';  
        fprintf('cate %d eig: time %fs\n', ci, toc(t1));
        %init_combo.init_Rc_arr{ci} = L2_normalization(init_combo.init_Rc_arr{ci}')';
        t2 = tic;
        tmp_mat = init_combo.init_Rc_arr{ci}*(Xcs*Xcs')*init_combo.init_Rc_arr{ci}';
        init_combo.init_Wc_arr{ci} = 1/param.cate_num*(train_label*Xcs'*init_combo.init_Rc_arr{ci}')/(tmp_mat+1e-10*mean(diag(tmp_mat))*eye(param.pca_dim)); % mat1*inv(mat2)
        fprintf('cate %d inverse: time %fs\n', ci, toc(t2));
        init_combo.init_Sc_arr{ci} = init_combo.init_Rc_arr{ci};
        init_combo.init_decs_arr{ci} = init_combo.init_Wc_arr{ci}*init_combo.init_Rc_arr{ci}*Xcs;

        init_combo.init_Wc_vals(ci) = norm(init_combo.init_Wc_arr{ci},'fro')^2;            
        init_combo.init_norm21_vals(ci) = calc_norm21(init_combo.init_Rc_arr{ci},param.K_per_cate);
        init_combo.init_RS_vals(ci) = norm(init_combo.init_Rc_arr{ci}*init_combo.init_Sc_arr{ci}','fro')^2;
        init_combo.init_diff_vals(ci) = norm(init_combo.init_Sc_arr{ci}'*init_combo.init_Sc_arr{ci}*Xcs-Xcs,'fro')^2;
        init_combo.init_mmd_vals(ci) = calc_mmd(init_combo.init_Rc_arr{ci},Xcs,Xct);
        init_combo.init_decs_sum = init_combo.init_decs_sum + init_combo.init_decs_arr{ci};
    end   
    
end
    