#include<iostream>
#include "math.h"
#include "mex.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)


double INF = 999999999999.0;

double max_vec(double *a, int n, int *mask, int &max_id)
{
    double max_val = -INF;
    for(int ii=0; ii<n; ii++)
    {
        if(a[ii]>max_val && mask[ii]!=-1)
        {
            max_val = a[ii];
            max_id = ii;
        }
    }
    return max_val;
}

double min_vec(double *a, int n, int *mask, int &min_id)
{
    double min_val = INF;
    for(int ii=0; ii<n; ii++)
    {
        if(a[ii]<min_val && mask[ii]!=-1)
        {
            min_val = a[ii];
            min_id = ii;
        }
    }
    return min_val;
}

double return_H(double *combo_kernel, double *bag_list, double *cate_list, double *domain_list, double *d_arr, int nd, int total_bag_num, int domain_num, int ii, int jj)
{
    int isample,jsample;
    double sum_val, H_ij;
    double tmp_val;
    int tmpi, tmpj;
    int icate, jcate, ik, jk, id;
    int d_bias = total_bag_num*(domain_num+1);
    int all_bias = d_bias*nd;
    
    // get the isample and jsample
    isample = (int)bag_list[ii]-1;
    jsample = (int)bag_list[jj]-1;
    icate = (int)cate_list[ii]-1;
    jcate = (int)cate_list[jj]-1;
    

    if(icate!=jcate)
    {
        return 0;
    }
    
    // calculate the dot product
    if(ii<total_bag_num && jj<total_bag_num)
    {
        sum_val = 0;
        for(id=0;id<nd;id++)
        {
            tmp_val = 0;
            for(ik=0; ik<domain_num; ik++)
            {
                tmpi = d_bias*id + isample*domain_num + ik;
                tmpj = jsample*domain_num + ik;
                tmp_val += combo_kernel[tmpi+tmpj*all_bias];
            }
            sum_val += d_arr[id]*tmp_val;
        }
    }
    else if(ii<total_bag_num && jj>=total_bag_num)
    {
        sum_val = 0;
        ik = (int)domain_list[jj]-1;
        for(id=0;id<nd;id++)
        {
            tmpi = d_bias*id + isample*domain_num + ik;
            tmpj = total_bag_num*domain_num + jsample;
            sum_val += d_arr[id]*combo_kernel[tmpi+tmpj*all_bias];
        }
    }
    else if(ii>=total_bag_num && jj<total_bag_num)
    {    
        sum_val = 0;
        ik = (int)domain_list[ii]-1;
        for(id=0;id<nd;id++)
        {
            tmpi = d_bias*id + jsample*domain_num + ik;
            tmpj = total_bag_num*domain_num + isample;
            sum_val += d_arr[id]*combo_kernel[tmpi+tmpj*all_bias];
        }
    }
    else
    {
        ik = (int)domain_list[ii]-1;
        jk = (int)domain_list[jj]-1;
        if(ik!=jk)
        {
            return 0;
        }     
        sum_val = 0;
        for(id=0;id<nd;id++)
        {
            tmpi = d_bias*id + total_bag_num*domain_num + isample;
            tmpj = total_bag_num*domain_num + jsample;
            sum_val += d_arr[id]*combo_kernel[tmpi+tmpj*all_bias];
        }
    }
    return sum_val;
}


mxArray* solve_qp_smo(const mxArray *combo_kernel_mx, const mxArray *bag_list_mx, const mxArray *cate_list_mx, const mxArray *domain_list_mx,\
        const mxArray *init_alpha_mx, const mxArray *init_neg_derive_mx, const mxArray *lb_mx, const mxArray *ub_mx, const mxArray *d_arr_mx,\
        const mxArray *total_bag_num_mx, const mxArray *domain_num_mx, const mxArray *max_iter_mx, const mxArray *eps_top_mx, const mxArray *quiet_mode_mx)
{   
    int ii,iter,ismpl;
    double top,bottom,max_top;
    int tmp_domain_idx;
    double min_domain_obj_val;
    int update_i,update_j;
    double tmp_lb,tmp_ub,tmp_a,tmp_b;
    double obj_val,delta_obj;
    double max_domain_diff;
    int tmpi,tmpj;
    double tmp_ss,tmp_tt,tmp_st;
    double min_delta_alpha;
    
    mxArray* presult = mxDuplicateArray(init_alpha_mx);
    double *alpha = mxGetPr(presult);
       
    double *pneg_derive = mxGetPr(init_neg_derive_mx);    
    double *lb = mxGetPr(lb_mx);
    double *ub = mxGetPr(ub_mx);
    double *combo_kernel = mxGetPr(combo_kernel_mx);
    double *bag_list = mxGetPr(bag_list_mx);  
    double *cate_list = mxGetPr(cate_list_mx);  
    double *domain_list = mxGetPr(domain_list_mx);  
    
    double *d_arr = mxGetPr(d_arr_mx);
    int nd = (int)mxGetM(d_arr_mx);
    
    int st_num = (int)mxGetM(bag_list_mx);
    int quiet_mode = (int)mxGetScalar(quiet_mode_mx);
        
    double *neg_derive = (double *)malloc(st_num*sizeof(double));
    for(ii=0;ii<st_num;ii++)
    {
        neg_derive[ii] = pneg_derive[ii];
    }
     
    // get parameters
    int SMO_MAX_ITER = (int)mxGetScalar(max_iter_mx);
    double eps_top = mxGetScalar(eps_top_mx);
    int total_bag_num = (int)mxGetScalar(total_bag_num_mx);
    int domain_num = (int)mxGetScalar(domain_num_mx);
        
    // prepare for choosing i
    double *domain_max_val = (double*)malloc(total_bag_num*sizeof(double));       
    int *domain_max_idx = (int*)malloc(total_bag_num*sizeof(int));
    
    // prepare for choosing j
    double *domain_obj_val = (double*)malloc(total_bag_num*sizeof(double));
    double *domain_diff = (double*)malloc(total_bag_num*sizeof(double));
    int *domain_min_idx = (int*)malloc(total_bag_num*sizeof(int));
       
    for(iter=0; iter<SMO_MAX_ITER; iter++)
    {
        // initialize
        for(int kk=0;kk<total_bag_num;kk++)
        {
            domain_max_val[kk] = -INF;
            domain_max_idx[kk] = -1;
            domain_obj_val[kk] = INF;
            domain_diff[kk] = -INF;
            domain_min_idx[kk] = -1;
        }    
            
        // choose i in working_set
        for(ii=0; ii<st_num; ii++)
        {
            if(alpha[ii]>=ub[ii])
            {
                continue;
            }
            ismpl = bag_list[ii]-1;
            if(neg_derive[ii]>domain_max_val[ismpl])
            {
                domain_max_val[ismpl] = neg_derive[ii];
                domain_max_idx[ismpl] = ii;
            }
        }

        // choose j in working_set
        for(ii=0; ii<st_num; ii++)
        {
            if(alpha[ii]<=lb[ii])
            {
                continue;
            }
            ismpl = bag_list[ii]-1;
            if(domain_max_idx[ismpl]==-1)
            {
                continue;
            }
            top = domain_max_val[ismpl]-neg_derive[ii];
            if(top<=0)
            {
                continue;
            }
            else if(top>domain_diff[ismpl])
            {
                domain_diff[ismpl] = top;
            }

            tmpi = domain_max_idx[ismpl];
            tmpj = ii;
            tmp_ss = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpi, tmpi);
            tmp_tt = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpj, tmpj);
            tmp_st = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpi, tmpj);
            bottom = tmp_ss + tmp_tt - tmp_st;
            obj_val = -pow(top,2)/bottom;
            
            if(obj_val<domain_obj_val[ismpl])
            {
                domain_min_idx[ismpl] = ii;
                domain_obj_val[ismpl] = obj_val;
            }
        }
        
        min_domain_obj_val = min_vec(domain_obj_val, total_bag_num, domain_min_idx, tmp_domain_idx);        
        
        update_i = domain_max_idx[tmp_domain_idx];
        update_j = domain_min_idx[tmp_domain_idx]; 

        tmp_lb = max(lb[update_i]-alpha[update_i],-(ub[update_j]-alpha[update_j]));
        tmp_ub = min(ub[update_i]-alpha[update_i],-(lb[update_j]-alpha[update_j]));
                
        tmpi = update_i;
        tmpj = update_j;
        tmp_ss = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpi, tmpi);
        tmp_tt = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpj, tmpj);
        tmp_st = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpi, tmpj);
        tmp_a = 1.0/2.0*(tmp_ss + tmp_tt - tmp_st);
        tmp_b = -neg_derive[update_i] + neg_derive[update_j];
        
        
        min_delta_alpha = -tmp_b/(2*tmp_a);
        min_delta_alpha = min(min_delta_alpha,tmp_ub);
        min_delta_alpha = max(min_delta_alpha,tmp_lb);
                
        double prev_alpha_i = alpha[update_i];
        double prev_alpha_j = alpha[update_j];
        alpha[update_i] = alpha[update_i]+min_delta_alpha;
        alpha[update_j] = alpha[update_j]-min_delta_alpha;
                
        delta_obj = tmp_a*pow(min_delta_alpha,2)+tmp_b*min_delta_alpha;

        max_domain_diff = max_vec(domain_diff, total_bag_num, domain_min_idx, tmp_domain_idx);
        max_top = max_domain_diff;
                    
        if (quiet_mode==0)
        {
            mexPrintf("smo_iter %d: %d %d delta_obj %.20f max_top %f\n", iter, update_i, update_j, delta_obj, max_top);
        }
        
        if(iter>0)
        {
            if(max_top<eps_top)
            {
                break;
            }
        }
        
        // update neg_derive
        for(ii=0;ii<st_num;ii++)
        {
            tmpi = ii;
            tmpj = update_i;
            tmp_ss = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpi, tmpj);
            tmpj = update_j;
            tmp_tt = return_H(combo_kernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, tmpi, tmpj);
            neg_derive[ii] -= (tmp_ss - tmp_tt)*min_delta_alpha;
        }
    }   
    
    free(domain_max_val);
    free(domain_max_idx);
    free(domain_obj_val);
    free(domain_diff);
    free(domain_min_idx);
    free(neg_derive);
    
    return presult;
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{   
    plhs[0] = solve_qp_smo(prhs[0], prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6], prhs[7], prhs[8], prhs[9], prhs[10], prhs[11], prhs[12], prhs[13]);
}	
