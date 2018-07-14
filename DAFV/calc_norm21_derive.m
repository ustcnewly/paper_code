function tR = calc_norm21_derive(R, K)
    R = R'; % after transpose, Ku*d
    d = size(R,2);
    u = size(R,1)/K;
    tR = reshape(R(:),u, K*size(R,2)); % u*Kd
    sum21 = 1./sum(tR.^2,1) ;
    tR = tR*sparse(1:length(sum21),1:length(sum21),sum21);
    tR = reshape(tR, K*u, d)';
    
end