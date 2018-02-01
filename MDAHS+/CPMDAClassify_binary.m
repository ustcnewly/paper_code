function [dv, model] = CPMDAClassify_binary(data,options)

Cs = options.Cs;
theta = options.theta;
gamma_A = options.gamma_A;
gamma_I = options.gamma_I;
LaplacianNorm = options.LaplacianNorm;
usebias = options.usebias;
verbose = 1;

assert(all( ismember(data.labels, [-1 0 1])==1 ));
nDomains = length(data.domain_indicator);
assert(nDomains==length(data.domain_data_index));
S = nDomains - 1;

source_id = find(data.domain_indicator==0);
target_id = find(data.domain_indicator==1);
assert(length(target_id)==1);
assert(length(source_id)==S);

yS = data.labels(cell2mat(data.domain_data_index(source_id)));
assert(isequal(unique(yS), [-1 1]'));

idx_t = data.domain_data_index{target_id};
nT = length(idx_t);

%== trained source classifiers
Fs = zeros(nT,S);
for s = 1 : S
    idx_s = data.domain_data_index{source_id(s)};
    clear Ks Kt
    Ks(:,:) = data.K(idx_s,idx_s, s); % choose the available feature
    ys = data.labels(idx_s);
    tmp_model = svmtrain(ys, [(1:size(Ks,1))',Ks],  sprintf('-t 4 -c %g -q', Cs));
    Kt = data.K(idx_t, idx_s, s);
    Fs(:,s) = Kt(:, tmp_model.SVs) * tmp_model.sv_coef - tmp_model.rho;
%     Fs(:,s) = Fs(:,s) * ys(1);
end


% compute weight
clear Ktt
Ktt = data.K(idx_t,idx_t,:);
Ktt = mean(Ktt, 3);
Dtt = sum(Ktt);
switch LaplacianNorm
    case 0
        Dtt = diag(Dtt);
        Ltt = Dtt - Ktt;
    case 1
        Dtt = 1 ./ sqrt(Dtt);
        Dtt = diag(Dtt);
        Ltt = eye(nT) - (Dtt*Ktt*Dtt);
    otherwise
        error('%s', mfilename);
end


NS = size(Fs,2);

[gamma, fval, exitflag, output] = quadprog( (Fs'*Ltt*Fs), zeros(NS,1), [], [], ones(1,NS), 1, zeros(NS,1), []);
assert(exitflag==1);

% training
if usebias==1
    Ktt = Ktt + 1; % add bias
end

y = Fs * gamma;

J = theta * eye(nT);
dual_alpha = (J*Ktt + gamma_A * theta * nT * eye(nT) + gamma_I * theta ./ nT * Ltt * Ktt) \ (J * y);

model.dual_alpha = dual_alpha;



% testing
dv = Ktt * dual_alpha;