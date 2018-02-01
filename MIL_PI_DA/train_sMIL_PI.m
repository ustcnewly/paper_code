function model = train_sMIL_PI(K, tK, y, p, svm_C, gamma, pos_bn, neg_fn)

n       = size(tK,1);
assert(pos_bn+neg_fn==n);

H       = 1/gamma*[tK tK; tK tK] + [K.*(y*y') zeros(n, n); zeros(n, 2*n)];
H       =  H + 1e-6*eye(size(H));
tmp     = -1/gamma*(tK*svm_C);
f       = [-p + tmp; tmp];
A1      = [];
b1      = [];
 
A2_rt   = [ones(1,pos_bn),zeros(1,neg_fn),ones(1,pos_bn),zeros(1,neg_fn)];
A2      = [y' zeros(1, n); A2_rt];  
b2      = [0; sum(svm_C(1:pos_bn))];     
lb      = zeros(2*n, 1);
ub      = [Inf(pos_bn,1);ones(neg_fn,1);Inf(n,1)];    

x       = quadprog(H, f, A1, b1, A2, b2, lb, ub);

alpha   = x(1:n);
    
lb = zeros(n,1);
ub = Inf(n,1);

ydf     = K*(alpha.*y) - p.*(y);
idx     = alpha>lb & alpha < ub;
if(sum(idx) == 0)
    setA    = ( (alpha<=lb)&(y<0) )|( (alpha>=ub)&(y>0) );
    setB    = ( (alpha>=ub)&(y<0) )|( (alpha<=lb)&(y>0) );
    uprho   = max(ydf(setA));
    lowrho  = min(ydf(setB));
    assert(uprho <= lowrho);
    b       = -0.5*(uprho+lowrho);    
else    
    b       = -sum(ydf(idx))/sum(idx);
end


model           = struct();
model.coef      = alpha.*y;
model.b         = b;
model.alpha     = alpha;