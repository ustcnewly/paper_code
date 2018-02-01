function model = train_sMIL_PI_DA(K, tK, Kss, Kst, y, p, svm_C1, svm_C2, gamma1, gamma2, B, epsilon, pos_bn, neg_fn)

n       = size(tK,1);
assert(pos_bn+neg_fn==n);

H       = [K.*(y*y')+1/gamma1*tK+1/gamma2*Kss, 1/gamma1*tK,     1/gamma2*Kss;
           1/gamma1*tK,                        1/gamma1*tK,     zeros(size(K)); 
           1/gamma2*Kss,                       zeros(size(K)),  1/gamma2*Kss];
H       = H + 1e-6*eye(size(H));  % avoid non-psd

ns      = size(Kst,1);
nt      = size(Kst,2);
assert(ns==n);

f1      = -p-1/gamma1*(tK*svm_C1)-1/gamma2*ns/nt*(sum(Kst,2).*svm_C2);
f2      = -1/gamma1*(tK*svm_C1);
f3      = -1/gamma2*ns/nt*(sum(Kst,2).*svm_C2);
f       = [f1; f2; f3];

       
A1_1    = [-ones(1,n)./svm_C2', zeros(1,n), -ones(1,n)./svm_C2'];       
A1_2    = [ones(1,n)./svm_C2', zeros(1,n), ones(1,n)./svm_C2'];
A1_3    = [ones(1,n)./svm_C2', zeros(1,n), ones(1,n)./svm_C2'];
A1      = [A1_1; A1_2; A1_3];
b1      = [ns*epsilon-n; n+ns*epsilon; B*n];
  
A2_1    = [y',zeros(1, 2*n)];
A2_2    = [ones(1,pos_bn),zeros(1,neg_fn),ones(1,pos_bn),zeros(1,neg_fn), zeros(1,n)];
A2      = [A2_1; A2_2];  
b2      = [0; sum(svm_C1(1:pos_bn))];     

lb      = zeros(3*n, 1);
ub      = [Inf(pos_bn,1);ones(neg_fn,1);Inf(n,1);Inf(n,1)];    
    
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