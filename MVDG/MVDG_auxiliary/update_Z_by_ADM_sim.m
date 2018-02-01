function [rgb_Z, dep_Z, obj_par] = update_Z_by_ADM_sim(rgb_W, dep_W, param)

mu = 0.1;
mu_max = 1e6;
mu_delta = 0.1;
ADM_epsilon = 1e-5;
ADM_MAX_ITER = 1000;

rgb_E = zeros(param.dim, param.pos_num);
dep_E = zeros(param.dim, param.pos_num);
rgb_Z = zeros(param.pos_num, param.pos_num);
dep_Z = zeros(param.pos_num, param.pos_num);
rgb_U = zeros(param.pos_num, param.pos_num);
dep_U = zeros(param.pos_num, param.pos_num);
rgb_P = zeros(param.dim, param.pos_num);
dep_P = zeros(param.dim, param.pos_num);

for ADM_iter = 1:ADM_MAX_ITER
    
    % Upate J
    svt_thred = param.lambda2/mu;
    
    [tmpU,tmpS,tmpV] = svd(rgb_Z+rgb_U/mu);
    tmpS = tmpS - diag(svt_thred*ones(size(tmpS, 1), 1));
    tmpS(tmpS < 0) = 0;
    rgb_J = tmpU*tmpS*tmpV'; 
    
    [tmpU,tmpS,tmpV] = svd(dep_Z+dep_U/mu);
    tmpS = tmpS - diag(svt_thred*ones(size(tmpS, 1), 1));
    tmpS(tmpS < 0) = 0;
    dep_J = tmpU*tmpS*tmpV';
    
    % Upate Z
    rgb_M = ((2*param.gamma+mu)*eye(param.pos_num)+mu*(rgb_W'*rgb_W))/(2*param.gamma);
    rgb_N = (mu*rgb_W'*(rgb_W-rgb_E)+mu*rgb_J+rgb_W'*rgb_P-rgb_U)/(2*param.gamma);
    
    dep_M = ((2*param.gamma+mu)*eye(param.pos_num)+mu*(dep_W'*dep_W))/(2*param.gamma);
    dep_N = (mu*dep_W'*(dep_W-dep_E)+mu*dep_J+dep_W'*dep_P-dep_U)/(2*param.gamma);
    
    rgb_Z = eye(param.pos_num)/(dep_M*rgb_M-eye(param.pos_num))*(dep_M*rgb_N+dep_N);
    dep_Z = eye(param.pos_num)/(rgb_M*dep_M-eye(param.pos_num))*(rgb_M*dep_N+rgb_N);
    
    % Update E
    rgb_E = (mu*(rgb_W-rgb_W*rgb_Z)+rgb_P)/(2*param.lambda11+mu);
    dep_E = (mu*(dep_W-dep_W*dep_Z)+dep_P)/(2*param.lambda11+mu);
    
    % Update U,P
    rgb_U = rgb_U+mu*(rgb_Z-rgb_J);
    rgb_P = rgb_P+mu*(rgb_W-rgb_W*rgb_Z-rgb_E);
    
    dep_U = dep_U+mu*(dep_Z-dep_J);
    dep_P = dep_P+mu*(dep_W-dep_W*dep_Z-dep_E);
    
    mu = min(mu*(mu_delta+1),mu_max);
   
    % break criterion
    if norm(rgb_W-rgb_W*rgb_Z-rgb_E,'inf')<ADM_epsilon && norm(dep_W-dep_W*dep_Z-dep_E,'inf')<ADM_epsilon &&...
            norm(rgb_Z-rgb_J,'inf')<ADM_epsilon && norm(dep_Z-dep_J,'inf')<ADM_epsilon
        break;
    end
    
end

[~,~,obj2,obj3] = calc_ADM_ori_obj(rgb_Z,dep_Z,rgb_W,dep_W,param);
obj_par = obj2+obj3;

    



