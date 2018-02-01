function z = zscorem(x, m0,s0)
s0(s0==0)=1;
z = ( x - repmat(m0, size(x,1), 1)) ./ repmat(s0, size(x,1), 1);