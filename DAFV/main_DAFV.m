function main_DAFV(data_combo, init_combo, param)

    config;
    % initilization
    Wc_arr   = init_combo.init_Wc_arr;
    Sc_arr   = init_combo.init_Sc_arr;
    Rc_arr   = init_combo.init_Rc_arr;
    decs_arr = init_combo.init_decs_arr;

    Wc_vals     = init_combo.init_Wc_vals;
    norm21_vals = init_combo.init_norm21_vals;
    mmd_vals    = init_combo.init_mmd_vals;
    RS_vals     = init_combo.init_RS_vals;
    diff_vals   = init_combo.init_diff_vals;
    decs_sum    = init_combo.init_decs_sum;
    
    train_idx = find(data_combo.domains==param.src_domain);
    test_idx = find(data_combo.domains==param.tgt_domain);
    train_label = data_combo.labels(:,train_idx);
    test_label = data_combo.labels(:,test_idx);
    test_label = sum(sparse(1:param.cate_num,1:param.cate_num,1:param.cate_num)*test_label,1);
    ns = length(train_idx);
    nt = length(test_idx);
    n = ns+nt;
    L = [1/(ns^2)*ones(ns), -1/(ns*nt)*ones(ns,nt); -1/(ns*nt)*ones(nt,ns), 1/(nt^2)*ones(nt)];
    H = eye(n)-1/n*ones(n);        

    for out_iter = 1:param.max_out_iter 
        param.out_iter = out_iter;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update W_c 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf('out_iter %d: Update W\n',out_iter);
        for in_iter = 1:param.max_in_iter
            for ci = 1:param.cate_num
                Xcs = data_combo.features{ci}(:,train_idx);
                Sc = Sc_arr{ci};          
                Wc_arr{ci} = (train_label-decs_sum+decs_arr{ci})*Xcs'*Sc'/(Sc*(Xcs*Xcs')*Sc'+param.lambda1*eye(param.pca_dim));                                        
                prev_decs = decs_arr{ci};
                decs_arr{ci} = Wc_arr{ci}*Sc_arr{ci}*Xcs;
                decs_sum = decs_sum+(decs_arr{ci}-prev_decs);
                Wc_vals(ci) = norm(Wc_arr{ci},'fro')^2;                                            
            end
            obj_Wc = 1/2*norm(decs_sum-train_label,'fro')^2 + param.lambda1/2*sum(Wc_vals);   
            fprintf('iter %d: obj_Wc %f\n', in_iter, obj_Wc);
            if in_iter>1 && abs(obj_Wc-prev_obj_Wc)<param.Wc_epsilon*abs(prev_obj_Wc)
                break;
            end
            prev_obj_Wc = obj_Wc;          
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update R_c 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        fprintf('out_iter %d: Update R\n',out_iter);
        for ci = 1:param.cate_num
            Xcs = data_combo.features{ci}(:,train_idx);
            Xct = data_combo.features{ci}(:,test_idx);
            Xc = [Xcs,Xct];
            Sc = Sc_arr{ci};
            t2=tic;
            eig_vec = calc_frac_eig(Xc*H*Xc', param.lambda2*Xc*L*Xc'-param.lambda5*(Sc'*Sc));
            fprintf('cate %d: eig time %fs\n',ci,toc(t2));
            Rc_arr{ci} = eig_vec(:,end-param.pca_dim+1:end)';
            Rc_arr{ci} = L2_normalization(Rc_arr{ci}')';

            RS_vals(ci) = norm(Rc_arr{ci}*Sc_arr{ci}','fro')^2;
            mmd_vals(ci) = calc_mmd(Rc_arr{ci},Xcs,Xct);
        end
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update S_c 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        fprintf('out_iter %d: Update S\n',out_iter);
        prev_Sc_obj = 1/2*norm(decs_sum-train_label,'fro')^2 + param.lambda3*sum(norm21_vals)...
            + param.lambda4/2*sum(diff_vals)- param.lambda5/2*sum(RS_vals);
        prev_Sc_arr = Sc_arr;
        prev_norm21_vals = norm21_vals;
        prev_RS_vals = RS_vals;                                        
        prev_diff_vals = diff_vals;
        prev_decs_arr = decs_arr;
        prev_decs_sum = decs_sum;
        fprintf('init_Sc_obj %.20f\n', prev_Sc_obj);    
        enable_drop = true;
        lr = param.init_lr;
        for in_iter = 1:param.max_in_iter                                                                     
            for ci = 1:param.cate_num
                % calculate the gradient
                Xcs = data_combo.features{ci}(:,train_idx);

                derive1 = Wc_arr{ci}'*((decs_sum-train_label)*Xcs');
                derive2 = param.lambda3*calc_norm21_derive(Sc_arr{ci}, param.K_per_cate);
                tmp_mat1 = (Sc_arr{ci}*Xcs)*Xcs';
                derive3 = param.lambda4*((tmp_mat1*Sc_arr{ci}')*Sc_arr{ci}...
                    + (Sc_arr{ci}*Sc_arr{ci}')*tmp_mat1 - 2*tmp_mat1);
                derive4 = -param.lambda5*(Sc_arr{ci}*Rc_arr{ci}')*Rc_arr{ci};
                derive = derive1+derive2+derive3+derive4;

                % learning rate
                Sc_arr{ci} = Sc_arr{ci}-lr*derive;

                norm21_vals(ci) = calc_norm21(Sc_arr{ci},param.K_per_cate);
                RS_vals(ci) = norm(Rc_arr{ci}*Sc_arr{ci}','fro')^2;                                        
                diff_vals(ci) = norm(Sc_arr{ci}'*(Sc_arr{ci}*Xcs)-Xcs,'fro')^2;
                prev_decs = decs_arr{ci};
                decs_arr{ci} = Wc_arr{ci}*Sc_arr{ci}*Xcs;
                decs_sum = decs_sum+(decs_arr{ci}-prev_decs);
            end   

            Sc_obj = 1/2*norm(decs_sum-train_label,'fro')^2 + param.lambda3*sum(norm21_vals)...
                    + param.lambda4/2*sum(diff_vals) - param.lambda5/2*sum(RS_vals);
            fprintf('in_iter %d: Sc_obj %.20f\n', in_iter, Sc_obj);
            if Sc_obj>prev_Sc_obj 
                if enable_drop==false
                    break;
                end
                %roll back
                Sc_arr = prev_Sc_arr;
                norm21_vals = prev_norm21_vals;
                RS_vals = prev_RS_vals;                                        
                diff_vals = prev_diff_vals;
                decs_arr = prev_decs_arr;
                decs_sum = prev_decs_sum;
                if lr<=param.min_lr
                    break;
                else
                    lr = lr*0.1;
                    fprintf('roll back, learning rate changed to %.20f\n', lr);
                    continue;
                end
            elseif abs(Sc_obj-prev_Sc_obj)/abs(prev_Sc_obj)<param.epsilon
                break;
            end
            enable_drop = false;
            prev_Sc_obj = Sc_obj;
        end
        % prediction
        param.step_flag = 'Sc';
        test_decs = zeros(param.cate_num, nt);
        for ci = 1:param.cate_num
           test_decs = test_decs + Wc_arr{ci}*Sc_arr{ci}*data_combo.features{ci}(:,test_idx);
        end
        [~,y_pred] = max(test_decs,[],1);
        [~,~,acc] = calc_confusion_matrix(y_pred, test_label);
      
    end 
end


