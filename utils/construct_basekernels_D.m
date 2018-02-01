function basekernels = construct_basekernels_D (kernel_types, kernel_params, D)
% ---------------------------------------------------------------------%
%INPUT:
%       -D             :  Distance matrix, L2 distance
%       -kernel_types   :{ 'rbf', 'lap', 'id',  'isd'}
%       -kernel_params  :{ }
%
%OUTPUT:
%       -basekernels    :M base kernel matrix
% ---------------------------------------------------------------------


[n1 n2] = size(D);
M = 0;
for i= 1:length(kernel_types)
    M = M + length(kernel_params{i});
end
basekernels = zeros(M, n1, n2);
cur = 1;
for i = 1:length(kernel_types)
    kernel_type = kernel_types{i};
    params = kernel_params{i};
    for j = 1:length(params)
        param = params(j);
        switch kernel_type
            case 'rbf'
                basekernels(cur,:,:) = exp(-param * D.^2);
            case 'lap'
                basekernels(cur,:,:) = exp(-sqrt(param) * D );
            case 'id'
                basekernels(cur,:,:) = 1./ (sqrt(param) * D + 1);
            case 'isd'
                basekernels(cur,:,:) = 1./ (param * D.^2 + 1);
        end
        cur = cur + 1;
    end
end
basekernels = real(basekernels);
