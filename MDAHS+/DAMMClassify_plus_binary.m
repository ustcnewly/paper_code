function [dv, model] = DAMMClassify_plus_binary(data,options)

target_usebias      = options.target_usebias;
source_usebias      = options.source_usebias;
use_weight          = options.use_weight;
theta               = options.theta;
verbose             = options.verbose;

assert(all( ismember(data.labels, [-1 0 1])==1 ));

nDomains = length(data.domain_indicator);
S = sum(data.domain_indicator==0);
assert(S == nDomains-1);
assert(nDomains==length(data.domain_data_index));


source_id = find(data.domain_indicator==0);
target_id = find(data.domain_indicator==1);
assert(length(target_id)==1);
assert(length(source_id)==nDomains-1);

yS = data.labels(cell2mat(data.domain_data_index(source_id)));
assert(isequal(unique(yS), [-1 1]'));
nS = length(yS);
nT = length(data.domain_data_index{target_id});
n = nS+nT;
idx_t = data.domain_data_index{target_id};

Fs = zeros(n,S);
for s = 1 : S
    clear Ks Kt
    idx_s = data.domain_data_index{source_id(s)};
    Ks(:,:) = data.K(idx_s,idx_s,s);
    ys = data.labels(idx_s);     
    tmp_model = rho_svmtrain_K(ys, Ks, options.C(s), source_usebias);  
    idx_st = [idx_s; idx_t];
    Kt(:,:) = data.K(idx_st, idx_s, s);  
    if source_usebias
        Fs(idx_st,s) = (Kt(:,tmp_model.SVs)+1) * tmp_model.sv_coef;   
    else
        Fs(idx_st,s) = Kt(:,tmp_model.SVs) * tmp_model.sv_coef;   
    end       
    s_models{s} = tmp_model;
end


KaddPrior = data.K;
for s = 1 : length(source_id)
    idx_s = data.domain_data_index{source_id(s)};
    idx_st = [idx_s; idx_t];
    if ~isinf(theta)
        KaddPrior(idx_st,idx_st,s) = KaddPrior(idx_st,idx_st,s) + (Fs(idx_st,s)*Fs(idx_st,s)')./theta;
    end
end

% assign the labels of the unlabeled data to zeros when training
y = data.labels;
y(data.domain_data_index{target_id}) = 0;
opt.usebias = target_usebias;
opt.verbose = verbose;
opt.p = options.p;
opt.C = options.C;
opt.br = options.br;
opt.out_max_iter = options.out_max_iter;
opt.in_max_iter = options.in_max_iter;
opt.lambda = options.lambda;

assert( all(opt.p==[2 1]) );

opt.cost_vec = zeros(n,1);
sample_per_domain = zeros(nDomains,1);
for s = 1 : nDomains
    sample_per_domain(s) = length(data.domain_data_index{s});
end

for s = 1 : nDomains
    idx = data.domain_data_index{s};
    if use_weight
        opt.cost_vec(idx) = opt.C(s) / length(idx) * max(sample_per_domain);
    else
        opt.cost_vec(idx) = opt.C(s);
    end   
end

data_tmp.K = KaddPrior;
if isfield(data,'Q')
    data_tmp.Q = data.Q;
end
data_tmp.domain_data_index = data.domain_data_index;
data_tmp.y = y;

for s = 1 : S   
    idx_s = data.domain_data_index{s};
    idx_t = data.domain_data_index{nDomains};
    idx_st = [idx_s; idx_t];
    if isinf(theta)
        data_tmp.X_new{s,1} = data.X_new{s};
    else
        data_tmp.X_new{s,1} = [data.X_new{s}, Fs(idx_st,s)./sqrt(theta)];
    end
end

model = DAMMC_cutting_plane_mb_sm_pi(data_tmp, opt);    

model.global_SVs = model.SVs;

gamma = zeros(S,1);
for s = 1 : S
    T = size(model.mu,2);
    idx_s = data.domain_data_index{s};
    idx_t = data.domain_data_index{nDomains};
    clear Kss Kst
    Kss(:,:) = data.K(idx_s(s_models{s}.SVs),  idx_s,s);
    Kst(:,:) = data.K(idx_s(s_models{s}.SVs),  idx_t,s);
    if source_usebias==1 && target_usebias == 1
        Kss = Kss + 1;
        Kst = Kst + 1;
    end
    alpha_s = model.alpha(idx_s);
    alpha_t = model.alpha(idx_t);
    for t = 1 : T        
        gamma(s) = gamma(s) + model.mu(s,t) * s_models{s}.sv_coef' * (Kss * (alpha_s.*model.y_set(idx_s, t)) + Kst * (alpha_t.*model.y_set(idx_t,t)));
    end
end
gamma = gamma./theta;
fprintf('gamma=%s\n', sprintf('%g\t', gamma));


assert(target_usebias==model.usebias);
dv = group_mkl_predict(KaddPrior(data.domain_data_index{target_id},:, :), data.domain_data_index, model);


