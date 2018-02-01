function [tpr, fpr] = calc_roc(gt, desc, m)
assert(all(~isnan(desc)))
assert(length(gt)==length(desc));


[desc, idx] = sort(desc, 'descend');
gt = gt(idx);

npos = sum(gt==1);
nneg = sum(gt==-1);


pos_idx = find(gt==1);
neg_idx = find(gt==-1);

dv_neg = desc(neg_idx);
dv_pos = desc(pos_idx);

step = max(1, floor(nneg/m));


m = min(m, nneg);
tpr = zeros(m,1);
fpr = zeros(m,1);

for i = 1 : m
    thr = dv_neg( (i-1)*step + 1);
    tpr(i) = sum(dv_pos>=thr)/npos;
    fpr(i) = sum(dv_neg>=thr)/nneg;
end