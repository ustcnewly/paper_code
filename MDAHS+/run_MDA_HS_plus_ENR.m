function run_MDA_HS_plus_ENR()

clear all; clc;

addpath('.\utils');
addpath('.\DAMMC_auxiliary');
addpath('.\tools\libsvm-dense-3.11\matlab');
addpath('c:\Program Files\Mosek\6\toolbox\r2009b\');

options.Cs = 1;
options.Ct = 10;
options.lambda = 10^5;
options.lambda_pi = 10^5;
options.theta = 0.01;

options.use_weight = 0;
options.source_usebias = 0;
options.target_usebias = 1;
options.br = 0:0.1:0.2;   
options.out_max_iter = 15;
options.in_max_iter = 50;

options.normalization_type_2D = 'l1';
options.normalization_type_3D = 'l1';

options.events = {'birthday';'parade';'picnic';'show';'sports';'wedding'};
options.dataset_dir = '.\Datasets';
options.domain_names = {'Google','Bing','Flickr','Kodak'};

options.Kernel_2D           = 'rbf';
options.KernelParam_2D      = 0;
options.Kernel_3D           = 'rbf';
options.KernelParam_3D      = 0;
options.verbose             = 1;

options.mkl_p               = [2 1];
options.featID_per_domain   = [1 0;1 0;0 1;1 1];

data = load_domain_data_text(options);
data = prepare_data_text(data, options);

result = DO_DAMMC_plus(data, options); 
fprintf('map %f\n', result.map);

