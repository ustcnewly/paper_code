function [info, model] = DAMMClassify(data,options)
%
% Inputs
%  -data
%   .S              : number of domains
%	.X  			: n x d feature matrix, with each row one sample
%	.labels			: n x 1 label vector
%	.domain_data_index  	: S x 1 cells, each cell stores the indice of the data belong to one domain
%	.domain_indicator	: S x 1 vector, 0 for source domain and 1 for target domain
%
%  -options
%	.C			: S x 1 vector, the cost (C in SVM) of each domain
%	.p 			: LpMKL's p
%   .rho        : minimum number of selected domains (>=1)
%	.source_usebias
%	.target_usebias
%
% Outputs
%  info
%	.overall_acc : overall accuarcy
%	.multiclass_acc 	: multiclass accuracy
%	.confusion_matrix : confusion matrix
%	.aps
%	.map
%  model: trained model
%

assert(length(data.domain_indicator)==length(data.domain_data_index));

source_id = find(data.domain_indicator==0);
target_id = find(data.domain_indicator==1);
assert(length(target_id)==1);
assert(length(source_id)>=1);

yS = data.labels(cell2mat(data.domain_data_index(source_id)));
nT = length(data.domain_data_index{target_id});

data_tmp = data; 
% 1-vs-all
classes = unique(yS);
Ft_pred = zeros(nT,length(classes));
aps = zeros(length(classes),1);
for k = 1 : length(classes)
    fprintf('Training class %d...\n', k);
    data_tmp.labels = 2*(data.labels==classes(k))-1;  
    yk_t = data_tmp.labels(data.domain_data_index{target_id});
    [Ft_pred(:,k), model.binary_models{k}] = DAMMClassify_binary(data_tmp, options);
    aps(k) = calc_ap(yk_t, Ft_pred(:,k));
    disp(aps);
end
[dv_pred, yT_pred] = max(Ft_pred, [], 2);
yT = data.labels(cell2mat(data.domain_data_index(target_id)));
info.overall_acc = mean(yT_pred(:)==yT(:));
info.gt = yT;
info.y_pred = yT_pred;
info.aps = aps;
info.map = mean(aps);
info.confusion_matrix = calc_confusion_matrix(yT_pred, yT);
info.multiclass_acc = mean(diag(info.confusion_matrix));
