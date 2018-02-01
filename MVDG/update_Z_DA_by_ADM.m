function [rgb_Z, dep_Z, obj_par] = update_Z_DA_by_ADM(rgb_W, dep_W, param)

mu = 0.1;
mu_max = 1e6;
mu_delta = 0.1;
ADM_epsilon = 1e-4;
ADM_MAX_ITER = 1000;

rgb_E = zeros(param.dim, param.pos_num);
dep_E = zeros(param.dim, param.pos_num);
rgb_Z = zeros(param.pos_num, param.pos_num);
dep_Z = zeros(param.pos_num, param.pos_num);
rgb_U = zeros(param.pos_num, param.pos_num);
dep_U = zeros(param.pos_num, param.pos_num);
rgb_V = zeros(param.pos_num, param.pos_num);
dep_V = zeros(param.pos_num, param.pos_num);
rgb_P = zeros(param.dim, param.pos_num);
dep_P = zeros(param.dim, param.pos_num);

rgb_S_coef = eye(param.pos_num)/(eye(param.pos_num)+rgb_W'*rgb_W);
dep_S_coef = eye(param.pos_num)/(eye(param.pos_num)+dep_W'*dep_W);

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
    
    % Update S
    rgb_S = 1/mu*rgb_S_coef*(mu*rgb_W'*(rgb_W-rgb_E)+mu*rgb_Z+rgb_W'*rgb_P+rgb_V);
    dep_S = 1/mu*dep_S_coef*(mu*dep_W'*(dep_W-dep_E)+mu*dep_Z+dep_W'*dep_P+dep_V);
    
    % Upate Z
    rgb_Q = 1/2*(rgb_J+rgb_S-1/mu*(rgb_U+rgb_V));
    dep_Q = 1/2*(dep_J+dep_S-1/mu*(dep_U+dep_V));
    rgb_Z = (param.gamma*mu*dep_Q+mu*(param.gamma+mu)*rgb_Q)/(mu*mu+2*param.gamma*mu);
    dep_Z = (param.gamma*mu*rgb_Q+mu*(param.gamma+mu)*dep_Q)/(mu*mu+2*param.gamma*mu);
    
    % Update E
    rgb_E = (mu*(rgb_W-rgb_W*rgb_S)+rgb_P)/(2*param.lambda11+mu);
    dep_E = (mu*(dep_W-dep_W*dep_S)+dep_P)/(2*param.lambda11+mu);
    
    % Update U,V,P
    rgb_U = rgb_U+mu*(rgb_Z-rgb_J);
    rgb_V = rgb_V+mu*(rgb_Z-rgb_S);
    rgb_P = rgb_P+mu*(rgb_W-rgb_W*rgb_S-rgb_E);
    
    dep_U = dep_U+mu*(dep_Z-dep_J);
    dep_V = dep_V+mu*(dep_Z-dep_S);
    dep_P = dep_P+mu*(dep_W-dep_W*dep_S-dep_E);
    
    mu = min(mu*(mu_delta+1),mu_max);
    
    % break criterion
    if norm(rgb_W-rgb_W*rgb_S-rgb_E,'inf')<ADM_epsilon && norm(dep_W-dep_W*dep_S-dep_E,'inf')<ADM_epsilon &&...
            norm(rgb_Z-rgb_J,'inf')<ADM_epsilon && norm(dep_Z-dep_J,'inf')<ADM_epsilon &&...
            norm(rgb_Z-rgb_S,'inf')<ADM_epsilon && norm(dep_Z-dep_S,'inf')<ADM_epsilon
        break;
    end

end

[~,~,obj2,obj3] = calc_ADM_ori_obj(rgb_Z,dep_Z,rgb_W,dep_W,param);
obj_par = obj2+obj3;





