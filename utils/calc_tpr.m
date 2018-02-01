function [tpr] = calc_tpr(gt, desc)
assert(all(~isnan(desc)));
assert(length(gt)==length(desc));

pos_index = find(gt==1);
tpr = mean(sign(desc(pos_index))==1);
