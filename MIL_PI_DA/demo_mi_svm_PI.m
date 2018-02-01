function demo_mi_svm_PI()

    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
    
    path.data_home = 'YOUR_DATA_PATH';
    
    fprintf('loading data....\n');
    load_decaf_feature = load(fullfile(path.data_home, 'visual_bags.mat'), 'pos_bags', 'neg_bags');
    load_text_feature = load(fullfile(path.data_home, 'text_bags.mat'), 'pos_bags', 'neg_bags');   
    load(fullfile(path.data_home,'test_labels.mat'), 'test_labels');
    load(fullfile(path.data_home,'test_features.mat'), 'test_features');
   
    param.max_bag_num = 60;
    param.bag_size = 5;
    param.tdim = 2000;
    param.svm_C = 0.1;
    param.gamma = 10;
    param.max_iter = 50;
          
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

    load(fullfile(path.data_home, 'test_features.mat'), 'test_features');
    visual_features = [pos_visual_features, neg_visual_features];
    text_features = [pos_text_features, neg_text_features];

    visual_features = L2_normalization(visual_features);
    test_features = L2_normalization(test_features);

    kernel_type = 'gaussian';

    vparam = struct();
    vparam.kernel_type = kernel_type;
    tparam = struct();
    tparam.kernel_type = 'linear';

    [kernel,kernel_param] = getKernel(visual_features, vparam);
    tkernel = getKernel(text_features, tparam);
    test_kernel = getKernel(test_features, visual_features, kernel_param);

    n = size(tkernel,1);
    mask = [ones(n/2,1);zeros(n/2,1)];
    tkernel = tkernel.*(mask*mask');

    libmodel = main_mi_SVM_PI(kernel,tkernel, pos_bags, neg_bags, param);
    coef                = zeros(size(kernel, 1), 1);
    coef(libmodel.SVs)  = libmodel.sv_coef;

    model = struct();
    model.coef          = coef*libmodel.Label(1);
    model.b             = -libmodel.rho*libmodel.Label(1);

    decs = test_kernel*model.coef + model.b;   
    ap  = calc_ap_k(test_labels, decs);
            
