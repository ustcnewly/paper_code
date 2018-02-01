function run_MDA_HS()

    addpath('.\utils');
    addpath('.\DAMMC_auxiliary');
    addpath('.\tools\libsvm-dense-3.11\matlab');
    
    options.Cs                  = 1;
    options.Ct                  = 10;
    options.theta               = 0.01;
        
    options.Kernel_2D           = 'rbf';
    options.KernelParam_2D      = 0;
    options.Kernel_3D           = 'rbf';
    options.KernelParam_3D      = 0;
    options.verbose             = 1;
    options.featID_per_domain   = [1,0;1,0;0,1;1,1];
    options.dataset_dir = '.\Datasets';

    options.domain_names = {'Google','Bing', 'Flickr', 'Kodak'};
    options.events = {'birthday'; 'parade'; 'picnic'; 'show'; 'sports'; 'wedding'};
    nor = 'l2';
    options.normalization_type_2D = nor;
    options.normalization_type_3D = nor;
    
    data = load_domain_data(options);
    data = prepare_data(data, options);

    options.mkl_p               = [2 1];
    options.source_usebias      = 1;
    options.target_usebias      = 1;
    options.use_weight          = 0;

    result = DO_MDA_HS(data, options);
    fprintf('map %f\n', result.map);
end
