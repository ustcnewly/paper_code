function out_param = xmy_accuracy(predict_val, tst_lbl)

[smpl_num, cate_num] = size(predict_val);
if min(size(tst_lbl)) >1
    assert(size(tst_lbl, 1) == smpl_num);
    assert(size(tst_lbl, 2) == cate_num);
    binary_cate_flag = 1;
else
    assert(length(tst_lbl) == smpl_num);
    binary_cate_flag = 0;
end
%assert(length(unique(tst_lbl)) == cate_num);

[predict_max, predict_lbl] = max(predict_val, [], 2);
%out_param.cate_prec = zeros(cate_num, 1);
out_param.cate_conf_mat = zeros(cate_num);
if binary_cate_flag
   tmp_num = 0;
   smpl_num = length(predict_lbl);
   for tmp_i = 1:smpl_num
       if 1 == tst_lbl(tmp_i, predict_lbl(tmp_i))
           tmp_num = tmp_num + 1;
       end
   end
   out_param.accuracy =  tmp_num/smpl_num;
   out_param.predict_lbl = predict_lbl;
else
    for cate_i = 1:cate_num
        cate_idx = (tst_lbl == cate_i);
        %  out_param.cate_prec(cate_i) = sum(predict_lbl(cate_idx) == cate_i)/sum(cate_idx);
        for cate_j = 1:cate_num
            out_param.cate_conf_mat(cate_i,cate_j) = sum(predict_lbl(cate_idx) == cate_j)/sum(cate_idx);
        end
    end
    %precision = mean(out_param.cate_prec);
    %precision = mean(diag(out_param.cate_conf_mat));
    %out_param.accuracy = precision;
    out_param.accuracy =  mean(diag(out_param.cate_conf_mat));
    out_param.predict_lbl = predict_lbl;
end
end