function [norm_feat, varargout] = func_normalization(feat, norm_type, varargin)
[n d] = size(feat);
switch norm_type  
    case 'l1'
        factor = sum(abs(feat), 2);
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);   
    case 'l2'
        factor = sqrt(sum(feat.^2, 2));
        factor(factor==0)=1;
        norm_feat = feat ./ repmat(factor, 1, d);       
    case 'zscore'
        if nargin > 2
            [norm_feat, param1, param2] = my_zscore(feat, varargin{1}, varargin{2});
        elseif nargin <= 2
            [norm_feat, param1, param2] = my_zscore(feat);
        end
        if nargout > 1
            varargout{1} = param1;
            varargout{2} = param2;
        end
    case 'raw'
        norm_feat = feat;
    case 'powscale'
        ppp = 0.3;
        norm_feat = feat;
        for i = 1:size(feat,1)
            norm_feat(i,:) = sign(feat(i,:)).*abs(feat(i,:)).^ppp;
        end
    case 'linscale'
        if nargin <= 2
            minvalue = min(fea,[],1);
            maxvalue = max(fea,[],1);
        else
            minvalue = varargin{1};
            maxvalue = varargin{2};
        end
        norm_feat = feat;
        for i = 1:size(feat,1)
            norm_feat(i,:) = (feat(i,:) - minvalue)./(maxvalue - minvalue);
        end
        if nargout > 1
            varargout{1} = minvalue;
            varargout{2} = maxvalue;
        end
    otherwise
        error('func_normalization');
end