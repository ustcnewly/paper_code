function [balanced_acc] = calc_balanced_acc(gt, desc)
% for test only
% gt = [-1 1 1 -1 -1];
% desc = [0 1 -1 0 0];

gt = gt(:);
gt(gt>0) = 1;
gt(gt<0) = -1;
desc = desc(:);
y = zeros(size(gt));
y(desc>=0) = 1;
y(desc<0) = -1;

TP = sum( gt == 1 & y == 1) / sum(gt == 1);
TN = sum( gt == -1 & y == -1) / sum(gt == -1);

balanced_acc = (TP+TN)/2;

%fprintf('Balanced loss = %f%%, TP = %f%%, TN = %f%%\n', balanced_loss*100, TP*100, TN*100);
end
