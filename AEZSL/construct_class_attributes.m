function new_class_attributes = construct_class_attributes(attr_types, attr_map, class_attributes, attrnorm_flag)
    new_class_attributes = [];
    for iattr = 1:length(attr_types)
        attr_indices =  attr_map(attr_types{iattr});
        if attrnorm_flag==1
            new_class_attributes = [new_class_attributes, L1_normalization(class_attributes(:, attr_indices)')'];
        else
            new_class_attributes = [new_class_attributes, class_attributes(:, attr_indices)];
        end
    end
end

