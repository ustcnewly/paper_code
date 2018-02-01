function obj = calc_obj(J,Qt,E,Ep,Ps,Pt,Xs,Xt,Z,Y1,Y2,Y3,Y4,mu,param)
    obj1 = calc_nuclear_norm(J)+calc_nuclear_norm(Qt);
    obj2 = param.lambda1*sum(sqrt(sum(E.^2,1)))+param.lambda0*sum(abs(Ep(:)));
    obj3 = sum(sum(Y1.*(Ps'*Xs-Pt'*Xt*Z-E)))+sum(sum(Y2.*(Ps-Pt-Ep)))+sum(sum(Y3.*(Z-J)))+sum(sum(Y4.*(Pt-Qt)));
    obj4 = mu/2*(norm(Ps-Pt-Ep,'fro')^2+norm(Z-J,'fro')^2+norm(Pt-Qt,'fro')^2+norm(Ps'*Xs-Pt'*Xt*Z-E,'fro')^2);    
    obj = obj1+obj2+obj3+obj4;
end

