function [eig_vec, eig_val] = calc_frac_eig(tmp1,tmp2)
    tmp_mat = (tmp1+1e-10*mean(diag(tmp1))*eye(size(tmp1)))\tmp2; % inv(mat1)*mat2
    [eig_vec, eig_val] = eig(tmp_mat);
    [~,sort_idx] = sort(real(diag(eig_val)),'descend');
    eig_vec = real(eig_vec(:,sort_idx));
    eig_val = eig_val(sort_idx);
end

