function model = main_MIL_CPB_PI(K, tK, data, param)

n   = size(K, 1);
assert(n == size(tK, 1));

bags    = data.bags;
labels  = data.labels;
y       = zeros(n, 1);
all_idx = [];
for bi = 1:length(bags)
    idx     = bags{bi};
    y(idx)  = ones(length(idx), 1)*labels(bi);
    all_idx = [all_idx; idx(:)]; 
end
assert(length(all_idx) == n);   
assert(length(unique(all_idx)) == n);  
y_set   = y;
maxIter     = 100;

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
pos_count = round(bag_size * param.pos_ratio);
y_pos = bit0(:, bitsum >= pos_count);
y_pos(y_pos<1) = -1;


for t = 1:maxIter
    % training classifier    
    [model, obj] = train_MIL_CPB_PI(y_set, K, tK, param);
    model.y_set     = y_set;
    model.y         = y_set*model.d;
    fprintf('Iter %d: obj = %f\n', t, obj);
    if(t>1 && obj > old_obj - 0.01*abs(old_obj))
        fprintf('Objective converges.\n');
        break;
    end
    if(t == maxIter)
         break;
    end
    old_obj = obj;
    
    % inference y
    y       = infer_y_enum(y_pos, model, bags, labels, K+1, y_set);
    y_set   = [y_set y]; %#ok<AGROW>
end

end
