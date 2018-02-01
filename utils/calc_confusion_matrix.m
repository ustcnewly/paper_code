function [conf, yset, acc] = calc_confusion_matrix(y_pred, y)
%
% [conf, yset, acc] = calc_confusion_matrix(y_pred, y)
%
yset = unique(y);
conf = zeros(length(yset));

for i = 1 : length(yset)
    idx = find(y==yset(i));
    conf(i,:) = hist(y_pred(idx), yset);
    conf(i,:) = conf(i,:)./length(idx);
end
%acc = mean(diag(conf));
acc = mean(y_pred == y);
