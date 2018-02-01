function [tnr] = calc_tnr(gt, desc)
assert(all(~isnan(desc)));
assert(length(gt)==length(desc));

neg_index = find(gt==-1);
tnr = mean(sign(desc(neg_index))==-1);
