% calc_accuracy can be used for multi-label
% label_set is label set, e.g., 1:nCate
% y_pred is the predicted label
% label{ci} is the label for category ci
    
label_set = 1:nCate;
conf = zeros(length(label_set));
for ci = 1 : length(label_set)
    idx = find(label{ci}==1);
    conf(ci,:) = hist(y_pred(idx), label_set);
    conf(ci,:) = conf(ci,:)./length(idx);
end

% acc for all categories
acc = diag(conf);

% average of acc
macc = mean(acc);

