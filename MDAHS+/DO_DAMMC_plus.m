function info = DO_DAMMC_plus(data, options)

data = expand_data_text(data, options);

usebias = options.source_usebias;
nDomains = length(data.domain_data_index);
S = nDomains-1;
if usebias
    for s = 1 : S
        idx_s = data.domain_data_index{s};
        idx_t = data.domain_data_index{nDomains};
        idx_st = [idx_s; idx_t];
        data.K(idx_st, idx_st,s) = data.K(idx_st, idx_st,s) +1;
    end

    options.source_usebias = 0;
end

feat_per_domain = options.featID_per_domain;
data.X_new = cell(S,1);
for s = 1 : S
    idx_s = data.domain_data_index{s};
    idx_t = data.domain_data_index{nDomains};
    idx_st = [idx_s; idx_t];
    f = find(feat_per_domain(s,:)==1);
    assert(length(f)==1);
    
    [U,A] = svd(data.K(idx_st,idx_st,s));
    tmp = sqrt(diag(A));      
    X_new = (U*diag(tmp)*U');
    data.X_new{s} = X_new;
end

options_tmp.verbose             = options.verbose;
options_tmp.theta               = options.theta;
options_tmp.C(1:S,1)            = options.Cs;
options_tmp.C(nDomains,1)       = options.Ct;
options_tmp.p                   = options.mkl_p;
options_tmp.use_weight          = options.use_weight;
options_tmp.source_usebias      = options.source_usebias;
options_tmp.target_usebias      = options.target_usebias;
options_tmp.theta               = options.theta;
options_tmp.br                  = options.br;
options_tmp.out_max_iter        = options.out_max_iter;
options_tmp.in_max_iter         = options.in_max_iter;
options_tmp.lambda              = options.lambda;

data.domain_indicator(1:S,1) = 0;
data.domain_indicator(nDomains,1) = 1;
[info, model] = DAMMClassify_plus(data, options_tmp);
info.model = model;