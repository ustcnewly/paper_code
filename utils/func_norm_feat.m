function norm_feat = func_norm_feat(feat, norm_type)
[n d] = size(feat);
switch norm_type
    case 'l1_zscore'
        factor = sum(abs(feat), 2);
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);
        norm_feat = my_zscore(norm_feat);
    case 'l1'
        factor = sum(abs(feat), 2);
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);
    case 'l2_zscore'
        factor = sqrt(sum(feat.^2, 2));
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);
        norm_feat = my_zscore(norm_feat);
    case 'l2'
        factor = sqrt(sum(feat.^2, 2));
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);
    case 'zscore_l1'
        feat = my_zscore(feat);
        factor = sum(abs(feat), 2);
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);
    case 'zscore_l2'
        feat = my_zscore(feat);
        factor = sqrt(sum(feat.^2, 2));
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);
    case 'zscore'
        norm_feat = my_zscore(feat);
    case 'raw'
        norm_feat = feat;
    case 'powscale'
        ppp = 0.3;
        norm_feat = feat;
        for i = 1:size(feat,2)
            norm_feat(:,i) = sign(feat(:,i)).*abs(feat(:,i)).^ppp;
        end
    case 'powscale_zscore'
        ppp = 0.3;
        norm_feat = feat;
        for i = 1:size(feat,2)
            norm_feat(:,i) = sign(feat(:,i)).*abs(feat(:,i)).^ppp;
        end
        norm_feat = my_zscore(norm_feat);
    case 'linscale'
        norm_feat = feat;
        for i = 1:size(feat,2)
            norm_feat(:,i) = (feat(:,i) - minvalue)./(maxvalue - minvalue);
        end
    case 'powscale_l2'
        norm_feat = func_normalization(feat, 'powscale');
        norm_feat = func_normalization(norm_feat, 'l2');
    otherwise
        error('func_norm_feat');
end