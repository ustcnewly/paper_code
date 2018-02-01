function model = main_sMIL_PI(K, tK, pos_bags, neg_bags, param)

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

K               = shrink_kernel(K, pos_count, 1, 1);
tK              = shrink_kernel(tK, pos_count, 1, 1);
labels          = [ones(pos_bn, 1); -ones(neg_fn, 1)];
svm_C           = param.svm_C*[ones(pos_bn, 1)/pos_bn; ones(neg_fn, 1)/neg_fn];
gamma           = param.gamma;

pos_ratio = param.rho;
p           = [(2*pos_ratio-1)*ones(pos_bn,1); ones(neg_fn, 1)];
bag_model   = train_sMIL_PI(K, tK, labels, p, svm_C, gamma, pos_bn, neg_fn);

% expand coef
coef        = bag_model.coef;
ins_coef    = [];
for bi = 1:pos_bn
    ins_coef = [ins_coef; (1/pos_count(bi))*coef(bi)*ones(pos_count(bi), 1)]; %#ok<AGROW>
end
ins_coef    = [ins_coef; coef(1+pos_bn:end)];
% expand alpha
model   = struct();
model.coef      = ins_coef;
model.b         = bag_model.b;

