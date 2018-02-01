function [alpha, beta, r, Rx, Ry, invRx, invRy, Sx, Sy, invSx, invSy] = kcca_reg(Kx, Ky, eta, kappa)
% KCCA function - regularized with kappa
%
% Performes Kernel Canoncial Correlation Analysis as a single symmetric eigenvalue problem with
% parital gram schmidt kernel decomposition
%
% Usage:
%   [nalpha, nbeta, r, Rx, Ry, Sx, Sy] = kcca_reg(Kx, Ky, eta, kappa)
%
%
% Input:
%   Kx, Ky - Kernel matrices corresponding to the two views,
%   eta    - precision parameter for gsd
%   kappa   - regularsation parameter (value 0 to 1)
%
% Output:
%   nalpha, nbeta - contains the canonical correlation vectors as columns
%   rc            - is a  vector with corresponding canonical correlations.
%   Rx, Ry    - Kx = Rx*Rx', Ky = Ry*Ry'
%   Sx, Xy    - Rx'*Rx + kappa*I = Sx*Sx', Ry'*Ry + kappa*I = Sy*Sy'
%

% instead of gsd one could use icd

if norm(mean(Kx))>1e-10
    % warning('Did you centre your data? It is better to center your data first.');
end
%disp('Decomposing Kernel with PGSO');
[Rx, RxSize, RxIndex] = gsd(Kx,eta);
[Ry, RySize, RyIndex] = gsd(Ky,eta);


if size(Rx,2) <= size(Ry,2)
    [alpha, beta, r, invRx, invRy, Sx, Sy, invSx, invSy] = kcca_reg_(Rx, Ry, kappa);
else
    [beta, alpha, r, invRy, invRx, Sy, Sx, invSy, invSx] = kcca_reg_(Ry, Rx, kappa);
end


function [alpha, beta, r, invRx, invRy, Sx, Sy, invSx, invSy] = kcca_reg_(Rx, Ry, kappa)
%disp('Creating new TxT matrix Z from MxT matrix R');
Zxx = Rx'*Rx;
Zxy = Rx'*Ry;
Zyy = Ry'*Ry;
Zyx = Zxy';
tEyeY = eye(size(Zyy));
tEyeX = eye(size(Zxx));

%disp('Computing nalpha eigenproblem');

Bx = (1-kappa)*tEyeX*Zxx+kappa*tEyeX;
Sx = chol(Bx)';
invSx = inv(Sx);
Ax = invSx*Zxy*inv((1-kappa)*tEyeY*Zyy+kappa*tEyeY)*Zyx*invSx';
Ax = 0.5*(Ax'+Ax)+eye(size(Ax,1))*10e-6;
[alpha_hat,rr] = eig(Ax);
rr = diag(rr);

By = (1-kappa)*tEyeY*Zyy+kappa*tEyeY;
Sy = chol(By)';
invSy = inv(Sy);


%disp('Sorting Output of nalpha');
r = sqrt(real(rr)); % as the original r we get is lamda^2
alpha_tilde = invSx'*alpha_hat; % as \hat{\alpha} = S'*\alpha - this find alpha tidal

% if you do not do the following line it means you will need to project your
% testing data into the Gram-Shmidt space.
invRx = Rx*inv(Rx'*Rx);
alpha = invRx*alpha_tilde;


%disp('Computing nbeta from nalpha');
% computing beta -- but as we cant comput the original beta
% we can only compute beta twidel and not the original beta
beta_tilde = inv((1-kappa)*tEyeY*Zyy+kappa*tEyeY)*Zyx*alpha_tilde;
t = size(Zyy,1);
beta_tilde = beta_tilde./repmat(r',t,1);

% again if you do not do the following line you must project your
% corresponding testing examples to gsd space
invRy = Ry*inv(Ry'*Ry);
beta = invRy*beta_tilde;


% Make sure the components are in order of magnitude
[r, i] = sort(r, 'descend');
alpha = alpha(:,i);
beta =	beta(:,i);
