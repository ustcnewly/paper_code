function demo_MIL_CPB_PI_DA()

    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
    
    path.data_home = 'YOUR_DATA_PATH';
    
    fprintf('loading data....\n');
    load_text_feature = load(fullfile(path.data_home, 'text_bags.mat'), 'pos_bags', 'neg_bags');
    load_decaf_feature = load(fullfile(path.data_home, 'visual_bags.mat'), 'pos_bags', 'neg_bags');
    load(fullfile(path.data_home,'test_labels.mat'), 'test_labels');
    load(fullfile(path.data_home,'test_features.mat'), 'test_features');
   
    param.max_bag_num = 60;
    param.bag_size    = 5;
    param.tdim = 2000;
    param.svm_C1 = 0.01;
    param.svm_C2 = 10^-5;
    param.gamma1 = 10;
    param.gamma2 = 100;
    param.B = 10^10;
    param.epsilon = 0;
    param.max_iter = 100;
        
    bag_num = min(length(load_decaf_feature.pos_bags), param.max_bag_num);

    pos_visual_features = [];
    pos_text_features = [];
    neg_visual_features = [];
    neg_text_features = [];

    for j = 1:bag_num        
        visual_feature = load_decaf_feature.pos_bags(j).features;   
        pos_visual_features = [pos_visual_features,visual_feature]; 
        visual_feature = load_decaf_feature.neg_bags(j).features;   
        neg_visual_features = [neg_visual_features,visual_feature];

        text_feature = load_text_feature.pos_bags(j).features(1:param.tdim,:);
        pos_text_features = [pos_text_features,text_feature];
        text_feature = load_text_feature.neg_bags(j).features(1:param.tdim,:);
        neg_text_features = [neg_text_features,text_feature];
    end

    visual_features = [pos_visual_features, neg_visual_features];
    visual_features(isnan(visual_features)) = 0;
    text_features = [pos_text_features,neg_text_features];

    bags = cell(0);
    for bi = 1:bag_num*2;
        bags{bi} = ((bi-1)*param.bag_size+1):bi*param.bag_size;
    end
    tmp_labels = [ones(bag_num, 1); -ones(bag_num, 1)];

    visual_features = L2_normalization(visual_features);
    new_test_features = L2_normalization(test_features);

    data.bags           = bags;
    data.labels         = tmp_labels;

    kparam.kernel_type = 'gaussian';
    [K, kernel_param]  = getKernel(visual_features, kparam);
    tkparam.kernel_type = 'linear';
    tK  = getKernel(text_features, tkparam);
    n   = size(K, 1);
    assert(n == size(tK, 1));

    Kss = K;
    Kst = getKernel(visual_features, new_test_features, kernel_param);
    test_kernel = Kst';

    mask_labels = [ones(n/2,1);zeros(n/2,1)];
    tK = tK.*(mask_labels*mask_labels');

    model = main_MIL_CPB_PI_DA(K, tK, Kss, Kst, data, param);

    idx = full(model.SVs);
    ay = model.sv_coef.*model.y(idx);                    
    decs = test_kernel(:, idx)*ay ;   
    ap  = calc_ap_k(test_labels, decs);
              