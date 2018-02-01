function demo_mi_svm_PI_DA()

    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
    
    path.data_home = 'YOUR_DATA_PATH';
    
    fprintf('loading data....\n');
    load(fullfile(path.data_home,'test_labels.mat'), 'test_labels');
    load(fullfile(path.data_home,'test_features.mat'), 'test_features');    
    load_decaf_feature = load(fullfile(path.data_home, 'visual_bags.mat'), 'pos_bags', 'neg_bags');
    load_text_feature = load(fullfile(path.data_home, 'text_bags.mat'), 'pos_bags', 'neg_bags');       
    
    param.max_bag_num = 60;
    param.bag_size = 5;
    param.tdim = 2000;
    param.svm_C1 = 0.01;
    param.svm_C2 = 10^-5;
    param.gamma1 = 10;
    param.gamma2 = 100;
    param.B = 10^10;
    param.epsilon = 0;

    bag_num = min(length(load_decaf_feature.pos_bags), param.max_bag_num);

    pos_visual_features = [];
    pos_text_features = [];
    pos_bags = [];

    for j = 1:bag_num        
        visual_feature = load_decaf_feature.pos_bags(j).features;   
        text_feature = load_text_feature.pos_bags(j).features(1:param.tdim,:);
        pos_visual_features = [pos_visual_features,visual_feature];
        pos_text_features = [pos_text_features,text_feature];

        bag = struct();
        bag.bag_size = param.bag_size;
        bag.visual_features = visual_feature;
        bag.text_features = text_feature;
        pos_bags = [pos_bags,bag];
    end

    neg_visual_features = [];
    neg_text_features = [];
    neg_bags = [];
    for j = 1:bag_num
        visual_feature = load_decaf_feature.neg_bags(j).features;   
        text_feature = load_text_feature.neg_bags(j).features(1:param.tdim,:);
        neg_visual_features = [neg_visual_features,visual_feature];
        neg_text_features = [neg_text_features,text_feature];

        bag = struct();
        bag.bag_size = param.bag_size;
        bag.visual_features = visual_feature;
        bag.text_features = text_feature;
        neg_bags = [neg_bags,bag];
    end

    visual_features = [pos_visual_features, neg_visual_features];
    visual_features = L2_normalization(visual_features);
    test_features = L2_normalization(test_features);

    vparam.kernel_type = 'gaussian';
    text_features = [pos_text_features, neg_text_features];
    [kernel,kernel_param] = getKernel(visual_features, vparam);
    tparam.kernel_type = 'linear';
    tkernel = getKernel(text_features, tparam);
    n = size(tkernel,1);
    mask = [ones(n/2,1);zeros(n/2,1)];
    tkernel = tkernel.*(mask*mask');

    Kss = kernel;
    Kst = getKernel(visual_features, test_features, kernel_param);
    test_kernel = Kst';       

    libmodel = main_mi_svm_PI_DA(kernel,tkernel, Kss, Kst, pos_bags, neg_bags, param);

    coef                = zeros(size(kernel, 1), 1);
    coef(libmodel.SVs)  = libmodel.sv_coef;

    model = struct();
    model.coef          = coef*libmodel.Label(1);
    model.b             = -libmodel.rho*libmodel.Label(1);

    decs = test_kernel*model.coef;    
    ap  = calc_ap_k(test_labels, decs);           
