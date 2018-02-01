function K = shrink_kernel(K, bag_count, row_start, col_start)
%
% K = shrink_kernel(K, bag_count, row_start, col_start)
%
% shrink the instance kernel into an mi-kernel, we always assume the row
% correspond to the instances of bags.
%
% NOTE that the original ins_K will be modified.
%
% Input:
%   ins_K       - N-by-M matrix, a kernel, N = pos_ins_num + neg_ins_num
%   bag_count   - B-by-1 vector, the size of each bag
%   row_start   - the start index to shrink
%   col-start   - the start index to shrink, only shrink row if col_start
%                 is empty
%
% by LI Wen, Oct 2, 2011

K = shrink_row(K, bag_count, row_start);
if(nargin>3)
    K = shrink_row(K', bag_count, col_start)';
end

%##################################################
% shrink_row
function kern = shrink_row(kern, bag_count, row_start)
pos_num     = sum(bag_count);

ins_count   = 0;
for bi = 1:length(bag_count)
    start   = row_start + ins_count;
    stop    = row_start + ins_count + bag_count(bi) - 1;
    kern(row_start+bi-1, :)   = sum(kern(start:stop, :), 1)/bag_count(bi);
    ins_count = ins_count + bag_count(bi);
end

% move rest elements up
rest_num = size(kern, 1) - (row_start+pos_num-1);
if(rest_num > 0)
    start   = row_start + length(bag_count);
    stop    = start + rest_num - 1;
    kern(start:stop, :) = kern(row_start+pos_num:end, :);
    kern    = kern(1:stop, :);
end