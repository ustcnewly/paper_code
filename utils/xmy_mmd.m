function kmm_weight = xmy_mmd(model, src_ftr, tgt_ftr, param)

tmp_file = [param.model_path '\' mmd_name(param) '.mat'];
%if false
if exist(tmp_file, 'file')
    load(tmp_file, 'kmm_weight');
    disp(['load mmd prior from: ' tmp_file]);
else
    cate_num = model.cate_num;
    kmm_weight = zeros(length(model.esvm.train_lbl), 1);
    
    for cate_i = 1:cate_num
        cate_idx = (model.esvm.train_lbl == cate_i);
        cate_smpl_num = sum(cate_idx);
        if 0 == cate_smpl_num
            disp(['train prior: no exemplar for cate: ' num2str(cate_i)]);
            continue;
        end
        cate_weights = model.esvm.esvm_weights(cate_idx, :);
        cate_bias = model.esvm.esvm_bias(cate_idx);
        %     if length(cate_idx) < size(trn_ftr, 1) && isfield(model.esvm, 'train_idx')
        %         disp('binary classification prior');
        %         cate_idx = model.esvm.train_idx;
        %     end
        cate_prior = mmd_prior(src_ftr, tgt_ftr, cate_weights, cate_bias, cate_idx, param, model);
        kmm_weight(cate_idx) = cate_prior;
        disp(['mmd cate: ' num2str(cate_i)]);
    end
    save(tmp_file, 'kmm_weight');
    disp(['save mmd prior to: ' tmp_file]);
end
end

function prob = mmd_prior(trn_ftr, tgt_ftr, cate_weights, cate_bias, cate_idx, param, model)
trn_smpl_num = size(trn_ftr, 1);
tgt_smpl_num = size(tgt_ftr, 1);
cate_smpl_num = sum(cate_idx);
min_val = double(eps('single'));

if 1 == param.mmd_raw
    src_prd_prob = trn_ftr;
    tgt_prd_prob = tgt_ftr;
elseif 0 == param.mmd_raw
    src_prd_val = cate_weights * trn_ftr' + repmat(cate_bias, 1, trn_smpl_num);
    tgt_prd_val = cate_weights * tgt_ftr' + repmat(cate_bias, 1, tgt_smpl_num);
    
    if 1 == param.predict_prob_flag
        src_prd_prob = exemplar_logistic_prob(src_prd_val)';
        tgt_prd_prob = exemplar_logistic_prob(tgt_prd_val)';
    elseif 2 == param.predict_prob_flag
        if isfield(model, 'sigmoid')
            src_prd_prob = exemplar_sigmoid_prob(src_prd_val, model.sigmoid)';
            tgt_prd_prob = exemplar_sigmoid_prob(tgt_prd_val, model.sigmoid)';
        else
            disp('train sigmoid for exmplar/target posterior');
        end
    else
        disp('need prob output for prior');
        return;
    end
    clear trn_ftr tgt_ftr src_prd_val tgt_prd_val
end

%%
if 0 == param.mmd_knl   %linear kernel
    knl1 = linear_kernel(src_prd_prob, src_prd_prob);
    knl2 = linear_kernel(tgt_prd_prob, tgt_prd_prob);
    knl12 = linear_kernel(src_prd_prob, tgt_prd_prob);
elseif 1 == param.mmd_knl %rbf kernel
    knl1 = rbf_kernel(src_prd_prob, src_prd_prob, param.mmd_sig);
    knl2 = rbf_kernel(tgt_prd_prob, tgt_prd_prob, param.mmd_sig);
    knl12 = rbf_kernel(src_prd_prob, tgt_prd_prob, param.mmd_sig);
else
    disp('unknown mmd kernel flag');
end

tgt_mean = mean(knl2(:));
neg_knl1 = knl1(~cate_idx,:);
neg_knl1 = neg_knl1(:,~cate_idx);
neg_mean1 = mean(neg_knl1(:));  %n-*n-
neg_knl12 = knl12(~cate_idx,:);
neg_mean12 = mean(neg_knl12(:)); %n-*tgt
pos_knl1 = knl1(cate_idx,:); 
pos_neg_knl1 = pos_knl1(:,~cate_idx); %n+*n-
pos_knl1 = pos_knl1(:, cate_idx); %n+*n+
pos_knl12 = knl12(cate_idx,:); %n+*tgt
pos_num = cate_smpl_num;
neg_num = trn_smpl_num - cate_smpl_num;

dist = zeros(cate_smpl_num, 1);
%cate_idx_ids = find(cate_idx);
for i = 1:cate_smpl_num
    %pos_id = cate_idx_ids(i);
    tmp_mean1 = pos_num * pos_num *pos_knl1(i,i) + 2*pos_num*sum(pos_neg_knl1(i,:)) + neg_num * neg_num * neg_mean1;
    tmp_mean1 = tmp_mean1/trn_smpl_num/trn_smpl_num;
    tmp_mean12 = sum(neg_mean12(:)) + pos_num * sum(pos_knl12(i,:));
    tmp_mean12 = tmp_mean12/trn_smpl_num/tgt_smpl_num;
    dist(i) = tmp_mean1 + tgt_mean - 2*tmp_mean12;
end
sig = param.mmd_sig2 * median(dist);
prob = exp(-dist/sig);
prob = prob/max(sum(prob), min_val);
end


function knl = rbf_kernel(ftr1, ftr2, sigma)
%input: ftr1: n1*d, ftr2: n2*d
%output: knl n1*n2
knl = L2_distance_2(ftr1', ftr2');
%div = 2*sigma*sigma;
div = sigma*median(knl(:));
knl = exp(-knl/div);
end

function knl = linear_kernel(ftr1, ftr2)
%input: ftr1: n1*d, ftr2: n2*d
%output: knl n1*n2
knl = ftr1*ftr2';
end

% function prob = mmd_prior(trn_ftr, tgt_ftr, cate_weights, cate_bias, cate_idx, param, model)
% trn_smpl_num = size(trn_ftr, 1);
% tgt_smpl_num = size(tgt_ftr, 1);
% cate_smpl_num = sum(cate_idx);
% min_val = double(eps('single'));
% src_prd_val = cate_weights * trn_ftr' + repmat(cate_bias, 1, trn_smpl_num);
% tgt_prd_val = cate_weights * tgt_ftr' + repmat(cate_bias, 1, tgt_smpl_num);
%
% if 1 == param.predict_prob_flag
%     src_prd_prob = exemplar_logistic_prob(src_prd_val);
%     tgt_prd_prob = exemplar_logistic_prob(tgt_prd_val);
% elseif 2 == param.predict_prob_flag
%     if isfield(model, 'sigmoid')
%         src_prd_prob = exemplar_sigmoid_prob(src_prd_val, model.sigmoid);
%         tgt_prd_prob = exemplar_sigmoid_prob(tgt_prd_val, model.sigmoid);
%     else
%         disp('train sigmoid for exmplar/target posterior');
%     end
% else
%     disp('need prob output for prior');
%     return;
% end
% clear trn_ftr tgt_ftr src_prd_val tgt_prd_val
%
% %%
% tgt_ave = sum(tgt_prd_prob, 2)/tgt_smpl_num;
% trn_sum_neg = sum(src_prd_prob(:, ~cate_idx),2);
% trn_mat = src_prd_prob(:, cate_idx);
% trn_ave_mat = (trn_mat * cate_smpl_num + repmat(trn_sum_neg,1,cate_smpl_num))/trn_smpl_num;
% dist = sum((trn_ave_mat - repmat(tgt_ave, 1, cate_smpl_num)).^2);
% sig = param.mmd_sig * median(dist);
% prob = exp(-dist/sig);
% prob = prob/max(sum(prob), min_val);
% end

function  nm = mmd_name(param)
src = [];
tgt = [];
for i = 1:length(param.slct_src_domain)
    src = [src num2str(param.slct_src_domain(i))];
end
for i = 1:length(param.slct_tgt_domain)
    tgt = [tgt num2str(param.slct_tgt_domain(i))];
end
nm = [ 'd' num2str(param.data_flag) ... %dataset
    ];
if isfield(param, 'bing_num')
    nm = [nm ...
        '_' num2str(param.bing_num) ... %zscore
        ];
end
nm =[nm ...
    'n' num2str(param.norm_flag) ... %normalization
    ];
if isfield(param, 'norm2_flag') && param.norm2_flag
    nm = [nm ...
        '_' num2str(param.norm2_flag) ... %zscore
        ];
end
if isfield(param, 'smpl_seed')
    nm = [nm ...
        'sd' num2str(param.smpl_seed) ... %zscore
        ];
end
if isfield(param,'zscore') && param.zscore
    nm = [nm ...
        'z' num2str(param.zscore) ... %zscore
        ];
end

nm = [nm ...
    's' num2str(src) ... %source domain
    't' num2str(tgt) ... %target domain
    'C' num2str(param.svm_C) ... %svm C
    'W' num2str(param.exemplar_weight) ... %exemplar weight
    'M' num2str(param.w_weight) ... %mu: W^2 weight
    'L' num2str(param.lambda1) '-' num2str(param.lambda2) ...%lambda
    'l' num2str(param.loss_flag) ...%loss flag
    'p' num2str(param.predict_prob_flag) ... %prob flag
    'm' num2str(param.memory) ... %memory
    ];
nm = [nm ...
    '_k' num2str(param.mmd_knl) ...
    'r' num2str(param.mmd_raw) ...
    ];
if isfield(param, 'mmd_sig') && param.mmd_knl
    nm = [nm ...
        'sg' num2str(param.mmd_sig) ...
        ];
end
if isfield(param, 'mmd_sig2')
    nm = [nm ...
        'sg' num2str(param.mmd_sig2) ...
        ];
end
nm = [nm ...
    '.mmd' ...
    ];
end