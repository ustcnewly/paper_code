function [ap] = calc_ap_k(gt, desc, k)
%   ap = calc_ap_k(gt, desc, k)
%
%   Calculate the "average precision" of top-k elems in a ranking result.
%   Modified from the function "calc_ap".
%
%   Input:
%       gt      -- ground truth lables
%       desc    -- decision values
%       k       -- top k, optional. absent or assign a value less than one,
%                  will get ap.
%   Output:
%       ap@k    -- Average Precision calculated on top-k elems.
%
% by LI Wen

gt = gt(:);
desc = desc(:);
[dv, ind] = sort(-desc); dv = -dv;

if(exist('k', 'var') && k>0 && length(ind) > k)
    ind = ind(1:k);
end

gt = gt(ind);
pos_ind = find( gt > 0 );
npos = length(pos_ind);
if npos == 0
    ap = 0;
else
    ap = mean( (1:npos)' ./ pos_ind(:) );
end
