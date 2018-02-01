#include "mex.h"
#include <omp.h>

void system_info();

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    int a[10],i, sum = 0;
    for (i = 0; i < 10; i++)
        a[i] = i;

    omp_set_dynamic(1);
    omp_set_num_threads(6);
#pragma omp parallel default(shared) private(i)
{
    mexPrintf("Max num threads %d.\n", omp_get_max_threads());
#pragma omp for
    for (i = 0; i < 10; i++)
    {
        sum += a[i];
        mexPrintf("Num threads %d, thread ID %d.\n", omp_get_num_threads(), omp_get_thread_num()); 
        mexPrintf("a[%d] = %d, sum = %d\n", i, a[i], sum);
    }
}
    mexPrintf("%d\n", sum);
	system_info();
}


void system_info()
{
	mexPrintf("sizeof(int)=%d\n", sizeof(int));
}
