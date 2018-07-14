clear;clc;

addpath('utils');
config;

%assert(mod(param.desc_dim,param.K_per_cate)==0);

data_combo = load(fullfile(param.home, 'data_combo.mat'));

% initialize variables
init_combo = init_DAFV(data_combo, param);

% main algorithm
main_DAFV(data_combo, init_combo, param);


                            