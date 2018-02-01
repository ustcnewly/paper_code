function kern = shrink_row(kern, bag_count, row_start)
%SHRINK_ROW Summary of this function goes here
%   instead shrink_kernel, we only shrink row here
%   this is used when calculating shrinked kernel_st

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

end

