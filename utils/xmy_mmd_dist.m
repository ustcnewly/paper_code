function dist = xmy_mmd_dist(ftr1, ftr2, param)
%ftr1: n*d; ftr2: m*d
if 0 == param.mmd_knl   %linear kernel
    ave1 = mean(ftr1);
    ave2 = mean(ftr2);
    tmp = ave1 - ave2;
    dist = tmp * tmp';
elseif 1 == param.mmd_knl %rbf kernel
    knl1 = rbf_kernel(ftr1, ftr1, param.mmd_sig);
    knl2 = rbf_kernel(ftr2, ftr2, param.mmd_sig);
    knl12 = rbf_kernel(ftr1, ftr2, param.mmd_sig);
    dist = mean(knl1(:)) + mean(knl2(:)) - 2*mean(knl12(:));
else
    disp('unknown mmd kernel flag');
end
end

function knl = rbf_kernel(ftr1, ftr2, sigma)
%input: ftr1: n1*d, ftr2: n2*d
%output: knl n1*n2
knl = L2_distance_2(ftr1', ftr2');
%div = 2*sigma*sigma;
div = sigma*median(knl(:));
knl = exp(-knl/div);
end