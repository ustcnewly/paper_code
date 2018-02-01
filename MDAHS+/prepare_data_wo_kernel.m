function data_out = prepare_data_wo_kernel(data)

% to avoid confusion and make the code clear, we put the four domains in
% the following order by default:
%           Google, Bing, YouTube, Kodak
% or        Google, Bing, Kodak, YouTube
assert(isequal(data.is_video, [0 0 1 1]'));
assert(isequal(data.feat_indicator_2D, [1 1 1 1]')); % video has 2D feature for the keyframe
assert(isequal(data.feat_indicator_3D, [0 0 1 1]'));
nDomains = length(data.domains);
S = nDomains-1;
assert(nDomains==4);


for s = 3 : 4
    keyframe_index = data.domains{s}.keyframe_index;
    data.domains{s}.feat_2D.x = zeros( length(keyframe_index), size(data.domains{s}.feat_2D.decaf,2));
    for i = 1 : length(keyframe_index)
        data.domains{s}.feat_2D.x(i,:) = mean(data.domains{s}.feat_2D.decaf(keyframe_index{i},:));
    end
    data.domains{s}.feat_3D.x = [
        data.domains{s}.feat_3D.idt_hog, ...
        data.domains{s}.feat_3D.idt_hof, ...
        data.domains{s}.feat_3D.idt_mbh];        
end

X = cell(nDomains,S);
X{1,1} = data.domains{1}.feat_2D.decaf;
X{2,2} = data.domains{2}.feat_2D.decaf;
X{3,3} = data.domains{3}.feat_3D.x;
X{4,1} = data.domains{4}.feat_2D.x;
X{4,2} = data.domains{4}.feat_2D.x;
X{4,3} = data.domains{4}.feat_3D.x;

Y = cell(nDomains,1);
Y{1} = data.domains{1}.labels;
Y{2} = data.domains{2}.labels;
Y{3} = data.domains{3}.video_labels;
Y{4} = data.domains{4}.video_labels;

data_out.X = X;
data_out.Y = Y;