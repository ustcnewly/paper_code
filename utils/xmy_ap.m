function out_param = xmy_ap(predict_val, tst_lbl)

[smpl_num, cate_num] = size(predict_val);
if min(size(tst_lbl)) >1
    assert(size(tst_lbl, 1) == smpl_num);
    assert(size(tst_lbl, 2) == cate_num);
    binary_cate_flag = 1;
else
    assert(length(tst_lbl) == smpl_num);
    binary_cate_flag = 0;
end

    cate_ap = zeros(cate_num, 1);
    if binary_cate_flag
        for cate_i = 1:cate_num
            cate_ap(cate_i) = myAP(predict_val(:,cate_i), tst_lbl(:, cate_i), 1);
        end
    else
        for cate_i = 1:cate_num
            cate_ap(cate_i) = myAP(predict_val(:,cate_i), tst_lbl, cate_i);
        end
    end
    out_param.ap = mean(cate_ap);
    out_param.cate_ap = cate_ap;
end