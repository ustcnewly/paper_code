#include <iostream>
#include "math.h"
#include "mex.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)

double INF = 999999999999.0;
double tau = 1e-10;

double get_Q_entry(double *Q_all, int ii, int jj, int pos_idx, int train_num, int pos_num)
{
    int inew,jnew;
    if(ii==0)
    {
        inew = pos_idx-1;
    }
    else
    {
        inew = ii-1+pos_num;
    }
    if(jj==0)
    {
        jnew = pos_idx-1;
    }
    else
    {
        jnew = jj-1+pos_num;
    }
    return Q_all[inew*train_num+jnew];
}
         
          
mxArray* solve_qp_smo(const mxArray *Q_all_mx, const mxArray *p_mx, const mxArray *lb_mx, const mxArray *ub_mx, const mxArray *init_derive_mx, const mxArray *init_alpha_mx, const mxArray *pos_idx_mx, const mxArray *pos_num_mx, const mxArray *max_iter_mx, const mxArray *eps_obj_mx, const mxArray *quiet_mode_mx)
{   
    int ii,iter,min_obj_idx;
    double a,b,tmpa,tmpb,tmp_lb,tmp_ub,delta_alpha,delta_obj,min_delta_alpha,min_delta_obj;
        
    int SMO_MAX_ITER = (int)mxGetScalar(max_iter_mx);
    int quiet_mode = (int)mxGetScalar(quiet_mode_mx);
    double eps_obj = mxGetScalar(eps_obj_mx);
    
    mxArray* presult = mxDuplicateArray(init_alpha_mx);
    double *alpha = mxGetPr(presult);
    double *lb = mxGetPr(lb_mx);
    double *ub = mxGetPr(ub_mx);
    double *Q_all = mxGetPr(Q_all_mx); 
    double *p = mxGetPr(p_mx); 
    double *p_derive = mxGetPr(init_derive_mx);   
    int nsample = (int)mxGetM(p_mx);
    int pos_num = (int)mxGetScalar(pos_num_mx);
    int train_num = pos_num+(nsample-1);

    int pos_idx = (int)mxGetScalar(pos_idx_mx);
    
    double *derive = (double *)malloc(nsample*sizeof(double));
    for(ii=0;ii<nsample;ii++)
    {
        derive[ii] = p_derive[ii];
    }

    for(iter=1; iter<SMO_MAX_ITER; iter++)
    { 
        // choose the updated sample
        min_delta_obj = INF;
        for(ii=0;ii<nsample;ii++)
        {
            tmpa = 1.0/2*get_Q_entry(Q_all, ii, ii, pos_idx, train_num, pos_num);
            tmpb = derive[ii];
            tmp_lb = lb[ii]-alpha[ii];
            tmp_ub = ub[ii]-alpha[ii];
            delta_alpha = max(-tmpb/(2*tmpa),tmp_lb);
            delta_alpha = min(delta_alpha,tmp_ub);
            delta_obj = tmpa*pow(delta_alpha,2)+tmpb*delta_alpha;
            if(delta_obj<min_delta_obj)
            {
                min_obj_idx = ii;
                min_delta_obj = delta_obj;
                min_delta_alpha = delta_alpha;
            }
        }
            
        //update alpha
        alpha[min_obj_idx] += min_delta_alpha;

        //update derive
        for(ii=0;ii<nsample;ii++)
        {
            derive[ii] += get_Q_entry(Q_all, ii, min_obj_idx, pos_idx, train_num, pos_num)*min_delta_alpha;
        }
        if(quiet_mode==0)
        {
            mexPrintf("Iter %d: %d delta_obj %.20f\n", iter, min_obj_idx, min_delta_obj);
        }
        if(abs(min_delta_obj)<eps_obj)
        {
            //mexPrintf("Iter %d: %d min_delta_obj %.20f eps_obj %.20f\n", iter, min_obj_idx, min_delta_obj,eps_obj);
            break;
        }
    }
    free(derive);

    return presult;
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{   
    plhs[0] = solve_qp_smo(prhs[0], prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6], prhs[7], prhs[8], prhs[9], prhs[10]);
}	
