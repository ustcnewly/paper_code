function save_test_results(save_file_name, var_val) %#ok<INUSD> 
    flag = 0;
    var_name = inputname(2);
    while flag==0 && ~exist(save_file_name, 'file')
        try
            eval(sprintf('%s=var_val', var_name));
            save(save_file_name, var_name, '-v7.3');
            flag = 1;
        catch exception
            disp(exception);
        end
    end
end

