1. Prepare 'data_combo.mat' file and put it in the folder "data". The data_combo.mat should contain the following matrices. Note that C is the number of categories, N is the number of samples, D is the number of domains, K is the number of Gaussian models in GMM per category, U is the dimension of local descriptors.
   

    * labels: CxN binary label matrix
    * domains: 1xN domain indices with each element in [1,...,D].
    * features: we train a GMM based on the training samples from each category and encode local descriptors based on totally C GMMs. So features has C cells with each cell containing a (KU)xN matrix.


2. Run "demo_run_DAFV.m".