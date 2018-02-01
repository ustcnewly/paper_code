function [acc] = calc_acc(gt, desc)
% acc = calc_acc(gt, desc)
%
% gt : ground truth labels [-1,1]
% desc : decision_values
%%%
assert(length(gt)==length(desc));
acc = mean(gt==sign(desc));
