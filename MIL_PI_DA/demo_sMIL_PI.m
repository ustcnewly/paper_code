function demo_sMIL_PI()

    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
    
    path.data_home = 'YOUR_DATA_PATH';
    
    fprintf('loading data....\n');
    load_visual = load(fullfile(path.data_home, 'visual_bags.mat'), 'pos_bags', 'neg_bags');
    txt = load(fullfile(path.data_home, 'text_bags.mat'), 'pos_bags', 'neg_bags');   
    load(fullfile(path.data_home,'test_labels.mat'), 'test_labels');
    load(fullfile(path.data_home,'test_features.mat'), 'test_features');
   
    param.max_bag_num = 60;
    param.bag_size = 5;
    param.tdim = 2000;
    param.svm_C = 100;
    param.gamma = 100;
    param.rho = 0.6;
    
    featuresA   = [];
    featuresB   = [];
    bag_num     = min(length(load_visual.pos_bags), param.max_bag_num);
    pos_bags    = [];
    for j = 1:bag_num
        bag     = struct();
        bag.bag_size    = param.bag_size;
        bag.features{1} = load_visual.pos_bags(j).features;
        bag.features{2} = txt.pos_bags(j).features(1:param.tdim,:);
        pos_bags    = [pos_bags bag];
        featuresA   = [featuresA bag.features{1}];
        featuresB   = [featuresB bag.features{2}];
    end

    neg_bags    = [];
    for j = 1:bag_num
        bag     = struct();
        bag.bag_size    = param.bag_size;
        bag.features{1} = load_visual.neg_bags(j).features;
        bag.features{2} = txt.neg_bags(j).features(1:param.tdim,:);
        neg_bags    = [neg_bags bag];
        featuresA   = [featuresA bag.features{1}];
        featuresB   = [featuresB bag.features{2}];
    end

    featuresA = L2_normalization(featuresA);
    test_features = L2_normalization(test_features);

    kernel_type = 'gaussian';
    vparam = struct();
    vparam.kernel_type  = kernel_type;
    tparam.kernel_type = 'linear';
    [kernel, kernel_param]  = getKernel(featuresA, vparam);
    test_kernel = getKernel(test_features, featuresA, kernel_param);     

    tkernel = getKernel(featuresB, tparam);
    mask = [ones(size(tkernel,1)/2,1);zeros(size(tkernel,1)/2,1)];
    tkernel = tkernel.*(mask*mask');

    model= main_sMIL_PI(kernel,tkernel, pos_bags, neg_bags, param);
    decs = test_kernel*model.coef;    
    ap = calc_ap_k(test_labels, decs);


