function Z = zscore(X)

% normalize data N*d 

M_X = mean(X);
S_X = std(X);
un  = ones(size(X,1),1);
Z   =(X-(un*M_X))./(un*S_X);

end
