[y,xt] = libsvmread('../heart_scale');
w = load('../heart_scale.wgt');
y(:) = 1;
xt = single(full(xt));
model=strain_weight(ones(length(y),1), y, xt, '-s 3');
% [l,a]=spredict(y, xt, model);

% xt = sparse(double(xt));
% model=train(y, xt);
% [l,a]=predict(y, xt, model);
