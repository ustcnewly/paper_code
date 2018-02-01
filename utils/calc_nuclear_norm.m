function val = calc_nuclear_norm(X)

    [~,S,~] = svd(X);
    val = sum(diag(S));
    
end
