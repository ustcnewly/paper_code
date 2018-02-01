#include "mex.h"
#include <omp.h>


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    const mxArray* mxX1 = prhs[0];
    const mxArray* mxX2 = prhs[1];
    
    int d = (int)mxGetM(mxX1);
    int n1 = (int)mxGetN(mxX1);
    int n2 = (int)mxGetN(mxX2);
    
    const mxLogical* X1 = (const mxLogical*)mxGetData(mxX1);
    const mxLogical* X2 = (const mxLogical*)mxGetData(mxX2);
    
    mxArray* mxD = mxCreateDoubleMatrix((mwSize)n1, (mwSize)n2, mxREAL);
    double* dists = mxGetPr(mxD);
    
    compute_hamming(X1, X2, dists, d, n1, n2);
    
    plhs[0] = mxD;
}



