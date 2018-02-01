#include<iostream>
#include "math.h"
#include "mex.h"

#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)


double INF = 999999999999.0;


double return_H_block(double *combo_kernel, double *bag_list, double *cate_list, double *domain_list, double *d_arr, int nd, int total_bag_num, int domain_num, int ii, int jj)
{
    int isample,jsample;
    double sum_val;
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


double return_H(double *combo_vkernel, double *combo_tkernel, double *bag_list, double *cate_list, double *domain_list, double *d_arr, int nd, int total_bag_num, int domain_num, int st_num, double gamma, int ii, int jj)
{
    
    double vval,tval,val;
    st_num = st_num/2;
    if(ii<st_num && jj<st_num)
    {
        vval = return_H_block(combo_vkernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, ii, jj);
        tval = return_H_block(combo_tkernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, ii, jj);
        val = vval+1.0/gamma*tval;
    }
    else if(ii<st_num && jj>=st_num)
    {
        jj = jj-st_num;
        tval = return_H_block(combo_tkernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, ii, jj);
        val = 1.0/gamma*tval;
    }
    else if(ii>=st_num && jj<st_num)
    {
        ii = ii-st_num;
        tval = return_H_block(combo_tkernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, ii, jj);
        val = 1.0/gamma*tval;
    }
    else
    {
        ii = ii-st_num;
        jj = jj-st_num;
        tval = return_H_block(combo_tkernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, ii, jj);
        val = 1.0/gamma*tval;
    }
    return val;
}
           
mxArray* calc_derive(const mxArray *combo_vkernel_mx, const mxArray *combo_tkernel_mx, const mxArray *bag_list_mx, const mxArray *cate_list_mx, const mxArray *domain_list_mx,\
        const mxArray *init_alpha_mx, const mxArray *f_mx, const mxArray *lb_mx, const mxArray *ub_mx, const mxArray *d_arr_mx,\
        const mxArray *total_bag_num_mx, const mxArray *domain_num_mx, const mxArray *gamma_mx)
{   
    int ii,jj;
    double H_ij;

    double *f = mxGetPr(f_mx);    
    double *init_alpha = mxGetPr(init_alpha_mx);
    double *combo_vkernel = mxGetPr(combo_vkernel_mx);
    double *combo_tkernel = mxGetPr(combo_tkernel_mx);
    double *bag_list = mxGetPr(bag_list_mx);  
    double *cate_list = mxGetPr(cate_list_mx);  
    double *domain_list = mxGetPr(domain_list_mx);  
    
    double *d_arr = mxGetPr(d_arr_mx);
    int nd = (int)mxGetM(d_arr_mx);
    int total_bag_num = (int)mxGetScalar(total_bag_num_mx);
    int domain_num = (int)mxGetScalar(domain_num_mx);
    double gamma = mxGetScalar(gamma_mx);
    int st_num = (int)mxGetM(bag_list_mx);
    
    mxArray* presult = mxCreateDoubleMatrix(st_num,1,mxREAL);
    double *result = mxGetPr(presult);
        
    for(ii=0;ii<st_num;ii++)
    {
        result[ii] = f[ii];
        for(jj=0;jj<st_num;jj++)
        {
            H_ij =  return_H(combo_vkernel, combo_tkernel, bag_list, cate_list, domain_list, d_arr, nd, total_bag_num, domain_num, st_num, gamma, ii, jj);
            result[ii] -= H_ij*init_alpha[jj];
        }
    } 
    return presult;
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{   
    plhs[0] = calc_derive(prhs[0], prhs[1], prhs[2], prhs[3], prhs[4], prhs[5], prhs[6], prhs[7], prhs[8], prhs[9], prhs[10], prhs[11], prhs[12]);
}	
