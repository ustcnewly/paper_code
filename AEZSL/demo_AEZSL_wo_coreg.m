clear; clc;

addpath('.\utils');
addpath('.\config');
addpath('.\graph_laplacian');

eval(['config_',dataset]);

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

Xtrain = features(:, trainInstancesIndices);
Xtest = features(:, testInstancesIndices);
XX = Xtrain*Xtrain';   

NN_ratio = 0.4;
GraphDistFun = 'euclidean';
GraphWeight = 'binary';   
lambda = 1;
sigma = 1;    

lap_opt.NN = ceil(NN_ratio*nTestClasses)-1;
lap_opt.GraphDistanceFunction = GraphDistFun;
lap_opt.GraphWeights = GraphWeight;
lap_opt.GraphWeightParam = 0;

S_all = adjacency(lap_opt,new_class_attributes);
Stest = S_all(unseen_classes,seen_classes);

param.lambda = lambda; 
param.sigma = sigma;

decs = zeros(nTestInstances,nTestClasses);
for ci = 1:nTestClasses
    S = full(diag(Stest(ci,:)));
    KYS = Xtrain*Y*(S*S')*A';
    KYS_invSS = KYS/(A*(S*S')*A'+param.sigma*eye(size(A,1)));
    Alpha=(XX+param.lambda*eye(size(XX,2)))\KYS_invSS;
    dec = (Atest(ci,:)*Alpha'*Xtest)';
    decs(:,ci) = dec;
end
[~, classPred] = max(decs,[],2);
classPred = unseen_classes(classPred);
acc = mean(testInstancesLabels==classPred);


