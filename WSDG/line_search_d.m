function d_arr = line_search_d(a,b)
    % use line search to find constant mid_val
    % and then calculate d_arr
    
    N = length(a);
    min_val = max(b);
    max_val = max(N^2*a+b);
    delta_val = max_val-min_val;
    MAX_ITER  = 1000000;
    iter = 0;
    while(abs(max_val-min_val)>min(1e-5,delta_val*1e-3))

        mid_val = (min_val+max_val)/2;
        if iter>0
            if prev_mid_val==mid_val
                break;
            end
        end
        sum_val = sum(a./sqrt(mid_val-b));
        if sum_val>1
            min_val = mid_val;
        elseif sum_val<1
            max_val = mid_val;
        else
            break;
        end
        
        iter = iter+1;
        if iter>MAX_ITER
            break;
        end
        
        prev_mid_val = mid_val;
    end
    if iter==0
        mid_val = (min_val+max_val)/2;
    end
    d_arr = a./sqrt(mid_val-b);
    d_arr(~isfinite(d_arr)) = 1;
    d_arr = d_arr/sum(d_arr);

end



