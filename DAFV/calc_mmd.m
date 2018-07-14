function result = calc_mmd(R, Xs, Xt)    
    result = norm(mean(R*Xs,2)-mean(R*Xt,2),'fro')^2;
end

