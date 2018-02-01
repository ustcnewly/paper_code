function data = expand_data(data, options)


feat_per_domain = options.featID_per_domain;

nDomains = length(data.domain_data_index);
S = nDomains - 1;
assert( any( sum(feat_per_domain(1:S,:),2)>=1 ) ); % at least one source domain has at least one type of feature
assert( all( feat_per_domain(nDomains,:)==any(feat_per_domain(1:S,:)==1)) ); % the target domain has all types of features

%%%%%%%%%
% convert the data struct to the setting that each source domain and each
% feature form a base kernel
%

[rr,cc] = find(feat_per_domain==1);
idx = rr~=nDomains;
rr = rr(idx); cc=cc(idx);
M = length(rr);
n = 0;
for s = 1 : S
    n = n + sum(feat_per_domain(s,:)==1) * length(data.domain_data_index{s});
end
nT = length(data.domain_data_index{nDomains});
n = n + nT;
K = zeros(n,n,M);

idx_t = (1:nT)' + (n-nT);

new_labels = zeros(n,1);
new_domain_data_index = cell(M+1,1);
offset_ = 0;
for m = 1 : M
    s = rr(m);
    f = cc(m);
    ns = length(data.domain_data_index{s});
    idx_s = (1:ns)' + offset_;
    new_domain_data_index{m} = idx_s;
    new_labels(idx_s) = data.labels(data.domain_data_index{s});
    offset_ = offset_ + ns;
    assert(feat_per_domain(s,f)==1);
    K(idx_s,idx_s, m) = data.K{s,s,f};
    K(idx_s,idx_t, m) = data.K{s,nDomains,f};
    K(idx_t,idx_s, m) = data.K{nDomains,s,f};    
    K(idx_t,idx_t, m) = data.K{nDomains, nDomains, f};
end
new_domain_data_index{M+1} = idx_t;
new_labels(idx_t) = data.labels(data.domain_data_index{nDomains});
data.K = K;
data.domain_data_index = new_domain_data_index;
data.labels = new_labels;

if isfield(data, 'c_ind')
    data = rmfield(data, 'c_ind');
end
if isfield(data, 'prepared_data_file')
    data = rmfield(data, 'prepared_data_file');
end