function demo_sMIL_PI_DA()

    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
    
    path.data_home = 'YOUR_DATA_PATH'; 
    
    fprintf('loading data....\n');
    load_visual = load(fullfile(path.data_home, 'visual_bags.mat'), 'pos_bags', 'neg_bags');
    txt = load(fullfile(path.data_home, 'text_bags.mat'), 'pos_bags', 'neg_bags');       
    load(fullfile(path.data_home,'test_labels.mat'), 'test_labels');
    load(fullfile(path.data_home,'test_features.mat'), 'test_features');
   
    param.bag_size = 5;
    param.max_bag_num = 60;
    param.tdim = 2000;    
    param.svm_C1 = 100;
    param.svm_C2 = 10^5;
    param.gamma1 = 100;
    param.gamma2 = 10^6;
    param.rho = 0.6;
    param.B = 10^10;
    param.epsilon = 0;    
    
    featuresA   = [];
    featuresB   = [];
    bag_num     = min(length(load_visual.pos_bags), param.max_bag_num);
    pos_bags    = [];
    for j = 1:bag_num
        bag     = struct();
        bag.bag_size    = param.bag_size;
        pos_bags    = [pos_bags bag]; 
        featuresA   = [featuresA load_visual.pos_bags(j).features]; 
        featuresB   = [featuresB txt.pos_bags(j).features(1:param.tdim,:)]; 
    end

    neg_bags    = [];
    for j = 1:bag_num
        bag     = struct();
        bag.bag_size    = param.bag_size;
        neg_bags    = [neg_bags bag]; 
        featuresA   = [featuresA load_visual.neg_bags(j).features]; 
        featuresB   = [featuresB txt.neg_bags(j).features(1:param.tdim,:)]; 
    end

    featuresA = L2_normalization(featuresA);
    test_features = L2_normalization(test_features);

    vparam.kernel_type = 'gaussian';
    tparam.kernel_type = 'linear';
    [kernel,kernel_param]  = getKernel(featuresA, vparam);
    tkernel = getKernel(featuresB, tparam);
    mask = [ones(size(tkernel,1)/2,1);zeros(size(tkernel,1)/2,1)];
    tkernel = tkernel.*(mask*mask');

    Kss  = kernel;
    Kst = getKernel(featuresA, test_features, kernel_param);
    test_kernel = Kst';

    model= main_sMIL_PI_DA(kernel,tkernel, Kss, Kst, pos_bags, neg_bags, param);        
    decs = test_kernel*model.coef;    
    ap  = calc_ap_k(test_labels, decs);          
