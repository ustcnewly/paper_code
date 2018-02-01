function d_mat = update_L21_SM(theta,wtile_norm)

    [T,M] = size(wtile_norm);
    d_mat_new = zeros(T,M);
    %% caculate the norm for each group; e_{t} = g_norm_{t}/sqrt(2*lambda) 
    g_norm = zeros(T,1);
    for tth = 1:T
        g_norm(tth,1) = sum(wtile_norm(tth,:).^(4/3)).^(3/4);
    end
    %% sorting g_norm
    [g_norm_sort, index_sort] = sort(g_norm,'descend');
    %% Find the number of the groups that have e_{t} to be equal to \theta;
    % according to the sorted elements of g_norm_sort;
    omega_s   = -1;
    sign_temp = 1;
    h_norm    = theta*ones(T,1);
    while (omega_s < T)&&(sign_temp)
       omega_s = omega_s + 1;%omega_s = 0,1,...,T-1;
       sum_sub = sum(g_norm_sort(omega_s+1:end,1));
       for hth = (omega_s + 1):T
           h_norm(hth,1) = (1-theta*omega_s)*g_norm_sort(hth,1)/sum_sub;
       end
       if h_norm(omega_s+1) < theta %Find the number;
           sign_temp = 0;
       end
    end
    %% caculate the new updated coefficients for each of the groups
    % step 1: for the groups that equals to theta;
    for s = 1:omega_s
        index_cur = index_sort(s);
        sum_temp_group = sqrt(sum(wtile_norm(index_cur,:).^(4/3)));
        for m = 1:M
            d_mat_new(index_cur,m) = theta*wtile_norm(index_cur,m).^(2/3)/sum_temp_group;
        end
    end
    % step 2: for the groups that is smaller to theta;
    sum_all_temp = sum(g_norm_sort(omega_s+1:T,1)); 
    if sum_all_temp>1e-6 % do not perform the computation for very small number;
        for s = omega_s+1:T
            index_cur = index_sort(s);
            sum_temp_group = (sum(wtile_norm(index_cur,:).^(4/3))).^(1/4);
            for m = 1:M
                d_mat_new(index_cur,m) = (1-theta*omega_s)*(wtile_norm(index_cur,m).^(2/3))*sum_temp_group/sum_all_temp;
            end
        end
    end
%%
    d_mat = d_mat_new; % update the coefficients    
end