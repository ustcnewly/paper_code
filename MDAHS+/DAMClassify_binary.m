function [dv, model] = DAMClassify_binary(data,options)
%
% Inputs
%  -data
%	.X  			: n x d feature matrix, with each row one sample
%	.labels			: n x 1 label vector [-1,1].  0 for target domain unlabeled data
%	.domain_data_index  	: S x 1 cells, each cell stores the indice of the data belong to one domain
%	.domain_indicator :
%   .Fs 			: decision values of source classifiers on the target data
%
%  -options
%   .init_d
%	.C			: scalar, the cost (C in SVM) of each domain
%	.theta
%   .svr_epsilon
%	.rho		: default 2
%	.beta		: default 1000
%
% Outputs
%  acc	: classification accuracy
%  model: trained model
%

Cs = 1;
Ct = options.C;
rho = options.rho;
beta = options.beta;
theta = options.theta;
svr_epsilon = options.svr_epsilon;
assert(all( ismember(data.labels, [-1 0 1])==1 ));
nDomains=length(data.domain_data_index);
S = nDomains-1;

source_id = find(data.domain_indicator(:)==0);
target_id = find(data.domain_indicator==1);
assert(length(target_id)==1);
assert(length(source_id)>=1);

yS = data.labels(cell2mat(data.domain_data_index(source_id)));
assert(isequal(unique(yS), [-1 1]'));
nT = length(data.domain_data_index{target_id});

idx_t = data.domain_data_index{target_id};
yT = data.labels(idx_t);

Fs = zeros(nT,S);
for s = 1 : S
    % decision values of the source classifiers on the target data
    idx_s = data.domain_data_index{source_id(s)};
    clear Ks Kt
    Ks(:,:) = data.K(idx_s,idx_s,s);
    ys = data.labels(idx_s);
    tmp_model = svmtrain(ys, [(1:size(Ks,1))',Ks], sprintf('-t 4 -c %g -q', Cs));
    Kt(:,:) = data.K(idx_t, idx_s,s);
    Fs(:,s) = Kt(:,tmp_model.SVs) * tmp_model.sv_coef - tmp_model.rho;
%     Fs(:,s) = Fs(:,s) * ys(1);
    clear Ks Kt idx_s ys
end

gamma = 1/S * ones(S,1);
gamma = gamma/sum(gamma);
gamma_tilde = gamma ./ sum(gamma);
clear Ktt
Ktt = data.K(idx_t,idx_t,:);
Ktt = mean(Ktt,3);

u_idx_t = find(yT==0);
l_idx_t = find(yT~=0);
II = ones(nT,1); II(u_idx_t)=1/sum(gamma);
II = diag(II);
hatK = Ktt + II./theta;
haty = Fs*(gamma_tilde);
haty(l_idx_t) = yT(l_idx_t);

model = svmtrain(haty, [(1:nT)', hatK], sprintf('-s 3 -t 4 -c %g -p %g -q', Ct, svr_epsilon));

dv = Ktt(:,model.SVs) * model.sv_coef - model.rho;


