function model = train_sMIL_PI_DA(K, tK, Kss,Kst, pos_bags, neg_bags, param)

pos_count    = zeros(length(pos_bags), 1);
pos_fn  = 0;
neg_fn  = 0;

for i = 1 : length(pos_bags)
    pos_fn  = pos_fn + pos_bags(i).bag_size;
    pos_count(i) = pos_bags(i).bag_size;
end
for i = 1 : length(neg_bags)
    neg_fn  = neg_fn + neg_bags(i).bag_size;
end

pos_bn  = length(pos_bags);

K = shrink_kernel(K, pos_count, 1, 1);
tK = shrink_kernel(tK, pos_count, 1, 1);
Kss = shrink_kernel(Kss, pos_count, 1, 1);
Kst = shrink_row(Kst, pos_count, 1);

labels          = [ones(pos_bn, 1); -ones(neg_fn, 1)];
svm_C1          = param.svm_C1*[ones(pos_bn, 1)/pos_bn; ones(neg_fn, 1)/neg_fn];
svm_C2          = param.svm_C2*[ones(pos_bn, 1)/pos_bn; ones(neg_fn, 1)/neg_fn];
gamma1          = param.gamma1;
gamma2          = param.gamma2;
B               = param.B;
epsilon         = param.epsilon;

pos_ratio = 0.6;
p           = [(2*pos_ratio-1)*ones(pos_bn,1); ones(neg_fn, 1)];
bag_model   = solve_qp_da(K, tK, Kss, Kst, labels, p, svm_C1, svm_C2, gamma1, gamma2, B, epsilon, pos_bn, neg_fn);

% expand coef
coef        = bag_model.coef;
ins_coef    = [];
for bi = 1:pos_bn
    ins_coef = [ins_coef; (1/pos_count(bi))*coef(bi)*ones(pos_count(bi), 1)];
end
ins_coef    = [ins_coef; coef(1+pos_bn:end)];

model   = struct();
model.coef      = ins_coef;
model.b         = bag_model.b;
