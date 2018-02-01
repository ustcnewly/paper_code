function data_tmp = prepare_data(data, options)



% to avoid confusion and make the code clear, we put the four domains in
% the following order by default:
%           Google, Bing, YouTube, Kodak
% or        Google, Bing, Kodak, YouTube
assert(isequal(data.is_video, [0 0 1 1]'));
assert(isequal(data.feat_indicator_2D, [1 1 1 1]')); % video has 2D feature for the keyframe
assert(isequal(data.feat_indicator_3D, [0 0 1 1]'));
S = length(data.domains);
assert(S==4);

Y = cell(S,1);
X = cell(S,2); % two types of features: 2D(decaf), 3D(HOG, HOF, MBH)

offset_ = 0;
for s = 1 : 2
    Y{s,1} =  data.domains{s}.labels;
    X{s,1} =  data.domains{s}.feat_2D.decaf;
    data_tmp.domain_data_index{s,1} = offset_ + (1 : length(Y{s}))'; 
    offset_ = offset_ + length(Y{s});
end

for s = 3 : S
    Y{s,1} = data.domains{s}.video_labels;
    X{s,1} = data.domains{s}.feat_2D.decaf;
    X{s,2} = [data.domains{s}.feat_3D.idt_hog, data.domains{s}.feat_3D.idt_hof, data.domains{s}.feat_3D.idt_mbh];
    data_tmp.domain_data_index{s,1} = offset_ + (1 : length(Y{s}))';
    offset_ = offset_ + length(Y{s});
end
data_tmp.labels = cell2mat(Y);
n = length(cell2mat(Y));

K = cell(S,S,2);
gamma0(1,1) = calc_g(cell2mat(X(1:2,1)));
gamma0(2,1) = calc_g(cell2mat(X(3:4,2)));


f = 1;
kernel_opt.Kernel = options.Kernel_2D;
kernel_opt.KernelParam = sqrt(0.5/gamma0(f));
%### 2D vs 2D
for i = [1 2]
    Xi = X{i,f};
    KK = calckernel(kernel_opt, Xi);
    K{i,i,f} = KK;
end


%### 2D vs 2D keyframe
tic
for i = 1 : 2
    Xi = X{i,f};
    for j =  4
        Xj = X{j,f};
        KK = calckernel(kernel_opt, Xj, Xi);
        keyframe_index = data.domains{j}.keyframe_index;
        KKK = zeros(size(KK,1), length(keyframe_index));
        for a = 1:length(keyframe_index)
            tmp = KK(:, keyframe_index{a});
            KKK(:,a) = mean(tmp, 2);
        end
        KK = KKK; clear KKK;
           
        K{i,j,f} = KK;
        K{j,i,f} = KK';
    end
end
toc


tic
%### 2D keyframe vs. 2D keyframe
for i = 4 : 4
      
    Xi = X{i,f};
    keyframe_index_i = data.domains{i}.keyframe_index;
    KK = zeros(length(keyframe_index_i), length(keyframe_index_i));
    for a = 1 : length(keyframe_index_i)
        KKK = calckernel(kernel_opt, Xi, Xi(keyframe_index_i{a},:));
        for b = 1 : length(keyframe_index_i)
            tmp = KKK(:,keyframe_index_i{b});
            KK(a,b) = mean(tmp(:));
        end
    end
    KK = (KK+KK')./2;
 
    K{i,i,f} = KK;   
end
toc

tic
f = 2;
kernel_opt.Kernel = options.Kernel_3D;
kernel_opt.KernelParam = sqrt(0.5/gamma0(f));

for i = 3:4
    Xi = X{i,f};
    KK = calckernel(kernel_opt, Xi);
    K{i,i,f} = KK;
    for j = i+1:4
        Xj = X{j,f};
        KK = calckernel(kernel_opt, Xj, Xi);
        K{i,j,f} = KK;
        K{j,i,f} = KK';
    end
end
toc
data_tmp.K = K;