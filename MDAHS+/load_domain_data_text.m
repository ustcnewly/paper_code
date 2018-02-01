function data_combo = load_domain_data_text(config)

    nDomains = length(config.domain_names);

    data_combo.domains = cell(nDomains,1);
    data_combo.domain_names = config.domain_names;
    
    dataset_dir = config.dataset_dir;
    
    for i = 1 : length(config.domain_names)
        fprintf('Loading %s ...\n', config.domain_names{i});
        switch config.domain_names{i}
            case {'Google'}
                load(fullfile(dataset_dir, 'Google', 'Google_combo_include_text.mat'), 'data');
                data_combo.domains{i} = data;
                assert(isequal(config.events, data_combo.domains{i}.events));
            case {'Bing'}
                load(fullfile(dataset_dir, 'Bing', 'Bing_combo_include_text.mat'), 'data');
                data_combo.domains{i} = data;
                assert(isequal(config.events, data_combo.domains{i}.events));
            case {'Kodak'}            
                load(fullfile(dataset_dir, 'Kodak', 'Kodak_combo.mat'), 'data');
                data_combo.domains{i} = data;
                assert(isequal(config.events, data_combo.domains{i}.events));
            case('Flickr')
                load(fullfile(dataset_dir, 'Flickr', 'Flickr_combo.mat'), 'data');
                data_combo.domains{i}  = data;  
            otherwise
                error('Wrong domain name: %s', config.domain_names{i});
        end    
    end

    data_combo.domains{i}.feat_indicator_2D = zeros(nDomains,1);
    data_combo.domains{i}.feat_indicator_3D = zeros(nDomains,1);
    data_combo.domains{i}.feat_indicator_text = zeros(nDomains,1);
    data_combo.domains{i}.is_video = zeros(nDomains,1);

    for i = 1 : length(data_combo.domains)
        if isfield(data_combo.domains{i}, 'feat_2D')
            if isfield(data_combo.domains{i}.feat_2D, 'decaf')
                data_combo.domains{i}.feat_2D.decaf = func_norm_feat(data_combo.domains{i}.feat_2D.decaf, config.normalization_type_2D);
            end
            data_combo.feat_indicator_2D(i,1) = 1;
        end

        if isfield(data_combo.domains{i}, 'feat_text')
            data_combo.domains{i}.feat_text = func_norm_feat(data_combo.domains{i}.feat_text,'l1');
            data_combo.feat_indicator_text(i,1) = 1;
        end

        if isfield(data_combo.domains{i}, 'feat_3D')
            if isfield(data_combo.domains{i}.feat_3D, 'idt_hof')
                data_combo.domains{i}.feat_3D.idt_hof = func_norm_feat(data_combo.domains{i}.feat_3D.idt_hof, config.normalization_type_3D);
            end
            if isfield(data_combo.domains{i}.feat_3D, 'idt_hog')
                data_combo.domains{i}.feat_3D.idt_hog = func_norm_feat(data_combo.domains{i}.feat_3D.idt_hog, config.normalization_type_3D);
            end
            if isfield(data_combo.domains{i}.feat_3D, 'idt_mbh')
                data_combo.domains{i}.feat_3D.idt_mbh = func_norm_feat(data_combo.domains{i}.feat_3D.idt_mbh, config.normalization_type_3D);
            end
            data_combo.feat_indicator_3D(i,1) = 1;
            data_combo.is_video(i,1) = 1;
        end

    end
end




