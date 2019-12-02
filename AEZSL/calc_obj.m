function obj = calc_obj(Xtrain, W_arr, A, Y, Stest, lambda, sigma, gamma)
    nTestClasses = length(W_arr);
    obj1 = 0;
    obj2 = 0;
    obj3 = 0;
    obj4 = 0;
    obj5 = 0;
    for ci = 1:nTestClasses
        S = full(diag(Stest(ci,:)));
        obj1 = obj1 + 1/2*norm((Xtrain'*W_arr{ci}*A-Y)*S,'fro')^2;
        obj2 = obj2 + lambda/2*norm(W_arr{ci}*A*S,'fro')^2;
        obj3 = obj3 + sigma/2*norm(Xtrain'*W_arr{ci},'fro')^2;
        obj4 = obj4 + lambda*sigma/2*norm(W_arr{ci},'fro')^2;
        for ci2 = ci+1:nTestClasses
            obj5 = obj5 + gamma/2*norm(W_arr{ci}-W_arr{ci2},'fro')^2;
        end
    end
    obj = obj1+obj2+obj3+obj4+obj5;
end