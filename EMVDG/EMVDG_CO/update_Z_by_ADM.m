function [Z_arr, obj_par] = update_Z_by_ADM(W_arr, param)

mu = 0.1;
mu_max = 1e6;
mu_delta = 0.1;
ADM_epsilon = 1e-4;
ADM_MAX_ITER = 1000;

E_arr = cell(param.feat_type_num,1);
Z_arr = cell(param.feat_type_num,1);
U_arr = cell(param.feat_type_num,1);
V_arr = cell(param.feat_type_num,1);
P_arr = cell(param.feat_type_num,1);
J_arr = cell(param.feat_type_num,1);
S_arr = cell(param.feat_type_num,1);
S_coef_arr = cell(param.feat_type_num,1);
for fti = 1:param.feat_type_num
    E_arr{fti} = zeros(param.dim, param.pos_num);
    Z_arr{fti} = zeros(param.pos_num, param.pos_num);
    U_arr{fti} = zeros(param.pos_num, param.pos_num);
    V_arr{fti} = zeros(param.pos_num, param.pos_num);
    P_arr{fti} = zeros(param.dim, param.pos_num);
    S_coef_arr{fti} = eye(param.pos_num)/(eye(param.pos_num)+W_arr{fti}'*W_arr{fti});
end

tmpL = -ones(param.feat_type_num, param.feat_type_num);
tmpL(logical(eye(param.feat_type_num))) = param.feat_type_num-1;

for ADM_iter = 1:ADM_MAX_ITER
   
    % Upate J
    svt_thred = param.lambda3/mu;    
    for fti = 1:param.feat_type_num
        [tmpU,tmpS,tmpV] = svd(Z_arr{fti}+U_arr{fti}/mu);
        tmpS = tmpS - diag(svt_thred*ones(size(tmpS, 1), 1));
        tmpS(tmpS < 0) = 0;
        J_arr{fti} = tmpU*tmpS*tmpV'; 
    end   
    
    % Update S
    for fti = 1:param.feat_type_num
        S_arr{fti} = 1/mu*S_coef_arr{fti}*(mu*W_arr{fti}'*(W_arr{fti}-E_arr{fti})+mu*Z_arr{fti}+W_arr{fti}'*P_arr{fti}+V_arr{fti});
    end
    
    % Upate Z
    H_vec = zeros(param.feat_type_num, param.pos_num*param.pos_num);
    for fti = 1:param.feat_type_num
        tmpH = 1/2*(J_arr{fti}+S_arr{fti}-1/mu*(U_arr{fti}+V_arr{fti}));
        H_vec(fti,:) = tmpH(:)';
    end
    Z_vec = mu*((param.gamma*tmpL+mu*eye(param.feat_type_num))\H_vec);
    for fti = 1:param.feat_type_num
        Z_arr{fti} = reshape(Z_vec(fti,:), param.pos_num, param.pos_num);
    end
   
    % Update E
    for fti = 1:param.feat_type_num
        E_arr{fti} = (mu*(W_arr{fti}-W_arr{fti}*S_arr{fti})+P_arr{fti})/(2*param.lambda2+mu);
    end
    
    % Update U,V,P
    for fti = 1:param.feat_type_num
        U_arr{fti} = U_arr{fti}+mu*(Z_arr{fti}-J_arr{fti});
        V_arr{fti} = V_arr{fti}+mu*(Z_arr{fti}-S_arr{fti});
        P_arr{fti} = P_arr{fti}+mu*(W_arr{fti}-W_arr{fti}*S_arr{fti}-E_arr{fti});
    end
    
    mu = min(mu*(mu_delta+1),mu_max);
    
    tmp_diffs = zeros(3, param.feat_type_num);
    for fti = 1:param.feat_type_num
        tmp_diffs(1,fti) = norm(W_arr{fti}-W_arr{fti}*S_arr{fti}-E_arr{fti});
        tmp_diffs(2,fti) = norm(Z_arr{fti}-J_arr{fti});
        tmp_diffs(3,fti) = norm(Z_arr{fti}-S_arr{fti});
    end
    % break criterion
    if all(tmp_diffs(:)<ADM_epsilon)
        break;
    end

end

[~,~,obj2,obj3] = calc_ADM_ori_obj(Z_arr,W_arr,param);
obj_par = obj2+obj3;





