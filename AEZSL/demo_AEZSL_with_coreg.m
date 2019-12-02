clear; clc;

addpath('.\utils');
addpath('.\config');
addpath('.\graph_laplacian');

eval(['config_',dataset]);

gamma = 1;
max_iter = 100;
eps = 1e-4;

param.lambda = 10^-3;
param.sigma = 10^-5;
lap_opt.NN = 99;
lap_opt.GraphDistanceFunction = 'cosine';
lap_opt.GraphWeights = 'binary';
lap_opt.GraphWeightParam = 0;


datasetPath=['..\..\dataset\',dataset];

load(fullfile(datasetPath,'data_combo.mat'), 'seen_classes', 'unseen_classes', 'image_attributes',...
    'image_classes', 'image_names', 'class_attributes');

trainInstancesIndices = find(ismember(image_classes, seen_classes));
testInstancesIndices = find(ismember(image_classes, unseen_classes));
trainInstancesLabels = image_classes(trainInstancesIndices);
testInstancesLabels = image_classes(testInstancesIndices);
nTrainingClasses = length(seen_classes);
nTrainingInstances = length(trainInstancesIndices);    
nTestClasses = length(unseen_classes);
nTotalClasses = nTrainingClasses+nTestClasses;
nTestInstances = length(testInstancesIndices);    

bias_flag = 1;
attrnorm_flag = 1;

new_class_attributes = construct_class_attributes(attr_types, attr_map, class_attributes, attrnorm_flag);
Atest = new_class_attributes(unseen_classes,:);               

load(fullfile(datasetPath,[feat_type,'.mat']), 'features');

feat_dim = size(features,1);
attr_dim = size(new_class_attributes,2);

Xtrain = features(:, trainInstancesIndices);
Xtest = features(:, testInstancesIndices);
XX = Xtrain*Xtrain';  
S_all = adjacency(lap_opt,new_class_attributes);
Stest = S_all(unseen_classes,seen_classes);


fprintf('svd XX....\n');
[UL, SL, ~] = svd(double(XX));
SL = diag(SL);
UR = cell(nTestClasses);
SR = cell(nTestClasses);
for ci = 1:nTestClasses
    S = full(diag(Stest(ci,:)));
    ASSA = A*(S*S')*A';
    fprintf('svd ASSA %d....\n',ci);
    [UR{ci}, SR{ci}, ~] = svd(double(ASSA));
    SR{ci} = diag(SR{ci});
end

% initialize W
init_sum_W = zeros(feat_dim, attr_dim);
init_W_arr = cell(nTestClasses,1);
for ci = 1:nTestClasses
    S = full(diag(Stest(ci,:)));
    KYS = Xtrain*Y*(S*S')*A';
    KYS_invSS = KYS/(A*(S*S')*A'+param.sigma*eye(size(A,1)));
    init_W_arr{ci} = (XX+param.lambda*eye(size(XX,2)))\KYS_invSS;
    init_sum_W = init_sum_W + init_W_arr{ci};
end

% main code
W_arr = init_W_arr;
sum_W = init_sum_W;

for iter = 1:max_iter
    for ci = 1:nTestClasses
        S = full(diag(Stest(ci,:)));
        tmpN = Xtrain*Y*(S*S')*A' + gamma*(sum_W-W_arr{ci});
        hatN = UL'*tmpN*UR{ci};
        Sigmas = (SL+param.lambda)*(SR{ci}+param.sigma)' + gamma*(nTestClasses-1);
        hatW = hatN./Sigmas;
        tmpW = UL*hatW*UR{ci}';
        sum_W = sum_W-W_arr{ci}+tmpW;
        W_arr{ci} = tmpW;
    end

    obj = calc_obj(Xtrain, W_arr, A, Y, Stest, param.lambda, param.sigma, gamma);
    fprintf('iter %d: obj %f\n', iter, obj);
    if iter>1 && abs(obj-prev_obj)/abs(prev_obj)<eps
        break;
    end
    prev_obj = obj;
end

    decs = zeros(nTestInstances,nTestClasses);
    for ci = 1:nTestClasses       
        decs(:,ci) = Xtest'*W_arr{ci}*Atest(ci,:)';
    end
    [~, classPred] = max(decs,[],2);
    classPred = unseen_classes(classPred);
    acc = mean(testInstancesLabels==classPred);
end



