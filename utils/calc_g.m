function mean_dis = calc_g(x)

ndata = size(x,2);

n2 = ones(ndata,1)*sum((x.^2), 1) + sum((x.^2),1)'*ones(1,ndata) - 2*(x'*x);
n2 = n2.*(1-eye(size(n2)));

mean_dis = mean(n2(:));
end