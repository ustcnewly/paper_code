function [auc] = calc_auc(gt,desc)
% %for test only
% gt = [-1 1 1 -1 -1];
% desc = [-1 1 1 -1 -1];

gt = gt(:);
desc = desc(:);
[dv, ind] = sort(-desc); dv = -dv;
gt = gt(ind);
neg_ind = find( gt < 0 );
pos_ind = find( gt > 0 );
npos = length(pos_ind);
nneg = length(neg_ind);
if npos == 0 || nneg == 0
    warning('pos = 0');
    auc = 0;
else
    auc = neg_ind(:) - (1:nneg)';
    auc = sum(auc) / npos / nneg;
end
end
