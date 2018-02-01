function [W,G,obj] = update_W_LRESVM(Z, G, pos_ftr, neg_ftr, param)

LRESVM_MAX_ITER = 100;

train_label = [ones(param.pos_num,1);-ones(param.neg_num,1)];
train_ftr = [pos_ftr; neg_ftr];

w_part2 = 1/(1+2*param.lambda1)*train_ftr'*diag(train_label);
Q_all = 1/(1+2*param.lambda1)*(train_ftr*train_ftr').*(train_label*train_label');
p0_all = 2*param.lambda1/(1+2*param.lambda1)*diag(train_label)*train_ftr; 
   
G_coef = param.lambda1*eye(param.pos_num)/(param.lambda2*(eye(param.pos_num)-Z)*(eye(param.pos_num)-Z)'+param.lambda1*eye(param.pos_num));

param.tmp_train_num = 1+param.neg_num;
lb = zeros(param.tmp_train_num,1);
ub = [param.C1;param.C2*ones(param.neg_num,1)];
loss_weight = [param.C1*ones(1,param.pos_num),param.C2*ones(1,param.neg_num)];

smo_max_iter = 100000;
smo_eps_obj = 1e-10;
quiet_mode = 1;   
init_alpha = zeros(length(lb),1);
loss_mask = [eye(param.pos_num);ones(param.neg_num,param.pos_num)];    
  
for iter = 1:LRESVM_MAX_ITER
    
    % Update W
    p_all = p0_all*G;
    w_part1 = 2*param.lambda1/(1+2*param.lambda1)*G;
    
    alpha_mat = zeros(param.train_num, param.pos_num);
    
    for ii = 1:param.pos_num       
        %('\titer %d %d/%d\n',iter, ii, param.pos_num);
        if param.year==2012
            alpha_mat([ii;(param.pos_num+1:param.pos_num+param.neg_num)'],ii) = solve_QP_SMO_allQ_12(Q_all, p_all([ii;(param.pos_num+1:param.pos_num+param.neg_num)'],ii)-ones(param.tmp_train_num,1), lb, ub, p_all([ii;(param.pos_num+1:param.pos_num+param.neg_num)'],ii)-ones(param.tmp_train_num,1), init_alpha, ii, param.pos_num, smo_max_iter, smo_eps_obj, quiet_mode);
        else
            alpha_mat([ii;(param.pos_num+1:param.pos_num+param.neg_num)'],ii) = solve_QP_SMO_allQ_14(Q_all, p_all([ii;(param.pos_num+1:param.pos_num+param.neg_num)'],ii)-ones(param.tmp_train_num,1), lb, ub, p_all([ii;(param.pos_num+1:param.pos_num+param.neg_num)'],ii)-ones(param.tmp_train_num,1), init_alpha, ii, param.pos_num, smo_max_iter, smo_eps_obj, quiet_mode);
        end
    end

    W = w_part1+w_part2*alpha_mat;

    % Update G
    G = W*G_coef;
    
    % calculate the objective
    obj = sum(loss_weight*max((1-diag(train_label)*train_ftr*W).*loss_mask,0)) + 1/2*trace(W'*W) + param.lambda2*norm(G-G*Z,'fro')^2 + param.lambda1*norm(G-W,'fro')^2;
    
    if iter>1 && (abs(obj-prev_obj)/abs(obj)<0.005 || obj>prev_obj)
        break;
    end
    prev_obj = obj;
    
    fprintf('\titer %d: obj %f\n', iter, obj);

end




