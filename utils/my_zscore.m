function [z, m, s] = my_zscore(x, m, s)

[n d] = size(x);
if nargin < 2
    m = mean(x);
    s = std(x);
    s(s==0)=1;
end

z = (x - repmat(m, n, 1))./repmat(s, n, 1);