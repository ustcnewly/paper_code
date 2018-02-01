function demo_MIL_CPB_PI()

    addpath('.\utils');
    addpath('.\tools\libsvm-3.17\matlab');
    addpath('C:\Program Files\Mosek\6\toolbox\r2009b');
    
    path.data_home = 'YOUR_DATA_PATH';
    
    fprintf('loading data....\n');
    load_text_feature = load(fullfile(path.data_home, 'text_bags.mat'), 'pos_bags', 'neg_bags');
    load_idt_feature = load(fullfile(path.data_home, 'visual_bags.mat'), 'pos_bags', 'neg_bags');
    load(fullfile(path.data_home,'test_labels.mat'), 'test_labels');
    load(fullfile(path.data_home,'test_features.mat'), 'test_features');
   
    param.max_bag_num = 60;
    param.bag_size = 5;
    param.tdim = 2000;
    param.svm_C = 0.01;
    param.gamma = 10;
    param.pos_ratio = 0.6;
    
    bag_num = min(length(load_idt_feature.pos_bags), param.max_bag_num);

    pos_visual_features = [];
    pos_text_features = [];
    neg_visual_features = [];
    neg_text_features = [];

    for j = 1:bag_num        
        visual_feature = load_idt_feature.pos_bags(j).features;   
        pos_visual_features = [pos_visual_features,visual_feature]; 
        visual_feature = load_idt_feature.neg_bags(j).features;   
        neg_visual_features = [neg_visual_features,visual_feature];

        text_feature = load_text_feature.pos_bags(j).features(1:param.tdim,:);
        pos_text_features = [pos_text_features,text_feature];
        text_feature = load_text_feature.neg_bags(j).features(1:param.tdim,:);
        neg_text_features = [neg_text_features,text_feature];
    end

    visual_features = [pos_visual_features, neg_visual_features];
    text_features = [pos_text_features,neg_text_features];

    bags = cell(0);
    for bi = 1:bag_num*2;
        bags{bi} = ((bi-1)*param.bag_size+1):bi*param.bag_size;
    end
    tmp_labels = [ones(bag_num, 1); -ones(bag_num, 1)];

    visual_features = L2_normalization(visual_features);
    new_test_features = L2_normalization(test_features);

    data.features       = visual_features;
    data.pi_features    = text_features;
    data.bags           = bags;
    data.labels         = tmp_labels;
    param.svm_C_star = 1;
    param.param.bag_size = param.bag_size;

    kparam = struct();
    kparam.kernel_type = 'gaussian';
    [K, kernel_param]  = getKernel(visual_features, kparam);
    test_kernel = getKernel(new_test_features, visual_features, kernel_param);
    tkparam.kernel_type = 'linear';
    tK  = getKernel(text_features, tkparam);
    mask_labels = [ones(size(tK,1)/2,1);zeros(size(tK,1)/2,1)];
    tK = tK.*(mask_labels*mask_labels');
    
    model = main_MIL_CPB_PI(K, tK, data, param);

    idx = full(model.SVs);
    ay = model.sv_coef.*model.y(idx);                    
    decs = test_kernel(:, idx)*ay ;              
    ap = calc_ap_k(test_labels, decs);
