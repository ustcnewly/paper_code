function D = calckernel_D(kernel_type, gamma, D)
% 
% K = calckernel(kernel_type, gamma, D),
%
%   D : L2 distance matri 
%   kernel_type : 'RBF', 'LAP', 'ISD', 'ID'
%
%   RBF:
%       K = exp(-gamma *  D.^2)
%   LAP:
%       K = exp(-sqrt(gamma) * D)
%   ISD:
%       K = 1 ./ (gamma * D.^2 + 1)
%   ID:
%       K = 1 ./  (sqrt(gamma) * D + 1)
%%%

switch upper(kernel_type)
    case 'RBF'
        D = exp(-gamma*D.^2);
    case 'LAP'
        D = exp(-sqrt(gamma)*D));
    case 'ISD'
        D = 1 ./ (gamma*D.^2 + 1);
    case 'ID'
        D = 1 ./ (sqrt(gamma)*D + 1);
    otherwise
        error('%s', mfilename);
end

