function model = main_mi_svm_PI_DA(K, tK, Kss, Kst, pos_bags, neg_bags, param)

pos_count    = zeros(1, length(pos_bags));
pos_fn  = 0;
neg_fn  = 0;
for i = 1 : length(pos_bags)
    pos_count(i) = pos_bags(i).bag_size;
    pos_fn = pos_fn + pos_bags(i).bag_size;
end
for i = 1 : length(neg_bags)
    neg_fn = neg_fn + neg_bags(i).bag_size;
end

labels = [ones(pos_fn, 1); -ones(neg_fn, 1)];
if(isfield(param, 'labels'))
    labels = param.labels;
end
        
param.pos_ratio = 0.2;
pos_lower_bound = ones(length(pos_bags), 1); % minimum pos ins in pos bags
if(isfield(param, 'pos_num'))
    pos_lower_bound = param.pos_num*pos_lower_bound;
elseif(isfield(param, 'pos_ratio'))
    pos_lower_bound = round(pos_count*param.pos_ratio);
end

param.max_iter = 3;
MAX_ITER = 50;
if(isfield(param, 'max_iter'))
    MAX_ITER = param.max_iter;
end

labels(1:pos_fn+neg_fn) = [ones(pos_fn, 1); -ones(neg_fn, 1)];
pos_bn  = length(pos_bags);

for iter = 1:MAX_ITER
    model   = train_mi_svm_PI_DA(labels, K, tK, Kss, Kst, param);
    if isempty(model.sv_coef)
        break;
    end

    if(iter>1 && norm(labels-labels_old)<sqrt(length(pos_bags)))
        break;
    end
        
    % predict label
    labels_old  = labels;    
    decs    = SVMTestKernel_libsvm(K(1:pos_fn, model.SVs), model);
    
    max_dec = max(decs(1:pos_fn));
    min_dec = min(decs(1:pos_fn));
    thresh = (max_dec-min_dec)*0.4 + min_dec;
    labels(1:pos_fn) = 2*(decs(1:pos_fn)>thresh)-1;  

    % check the bag constriants
    ins_count = 0;
    for bi = 1:pos_bn
        start   = ins_count + 1;
        stop    = ins_count + pos_count(bi);
        
        [~,ind] = sort(decs(start:stop), 'descend');
        labels(start + (ind(1:pos_lower_bound(bi))) - 1) = 1;
        if sum(labels(start:stop) == 1) == 0
            [~,mid] = max(decs(start:stop)); 
            labels(start+mid-1) = 1;
        end
        ins_count = ins_count + pos_count(bi);
    end
end
end

function decs = SVMTestKernel_libsvm(kernel, model)

decs    = kernel*model.sv_coef - model.rho;
decs    = decs*model.Label(1);
end