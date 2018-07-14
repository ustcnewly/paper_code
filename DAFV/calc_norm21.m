function result = calc_norm21(R, K)
    R = R'; % after transpose, Ku*d
    u = size(R,1)/K;
    tR = reshape(R(:),u, K*size(R,2)); % u*Kd
    result = sqrt(sum(tR.^2,1));
    result = sum(result);
end