function model = main_MIL_CPB_PI_DA(K, tK, Kss, Kst,data, param)

maxIter     = param.max_iter;
tau         = 0.001;

bags    = data.bags;
labels  = data.labels;
n       = size(K,1);
y       = zeros(n, 1);
all_idx = [];
for bi = 1:length(bags)
    idx     = bags{bi};
    y(idx)  = ones(length(idx), 1)*labels(bi);
    all_idx = [all_idx; idx(:)]; %#ok<AGROW>
end
assert(length(all_idx) == n);   
assert(length(unique(all_idx)) == n);   
y_set   = y;

% generate qualified y set
bag_size = param.bag_size;
bit0 = [0; 1];
for i = 2 : bag_size
    half_len = size(bit0, 1);
    bit0 = [zeros(half_len, 1), bit0; ones(half_len, 1), bit0]; %#ok<AGROW>
end;
bit0 = bit0';
% remove all y's which do not meet the requirement
bitsum = sum(bit0, 1);
param.pos_ratio = 0.2;
pos_count = round(bag_size * param.pos_ratio);
y_pos = bit0(:, bitsum >= pos_count);
y_pos(y_pos<1) = -1;

for t = 1:maxIter
    [model, obj] = train_MIL_CPB_PI_DA(y_set, K, tK, Kss, Kst, param);
    model.y_set     = y_set;
    model.y         = y_set*model.d;
    fprintf('Iter %d: obj = %f\n', t, obj);
    
    if(t>1 && obj > old_obj - tau*abs(old_obj))
        fprintf('Objective converges.\n');
        break;
    end

    if(t == maxIter)
        fprintf('Reach maximum number of iterations (%d).\n', maxIter);
        break;
    end
    old_obj = obj;
    
    % inference y
    y       = infer_y_enum(y_pos, model, bags, labels, K+1, y_set);
    y_set   = [y_set y]; %#ok<AGROW>
end

end

