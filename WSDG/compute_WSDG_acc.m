function [acc, y_pred, max_decs] = compute_WSDG_acc(test_decs, label, param)
% compute accuracy based on combo decs
  
    max_decs = zeros(param.test_num,param.cate_num);
    for ii = 1:param.test_num
        for ci = 1:param.cate_num
            max_decs(ii,ci) = max(test_decs(ii,(ci-1)*param.domain_num+1:ci*param.domain_num));
        end
    end
    
    [~,y_pred] = max(max_decs,[],2);
    [~,~,acc] = calc_confusion_matrix(y_pred, label);
    
end

