% Computes and saves the operator H_k acting on the Fourier basis 
% for a collection of grid points in the k-plane. 
% H_k is the following integral operator:
%
%   (H_k f)(x) = \int_{unitcircle} Htilde(k*(x-y))*f(y) dsigma(y), 
%
% where x is a point on the unit circle, 
% Htilde(x) = H1(x)-H1(0) = G1(x)-G0(x)-H1(0),
% and G1 is Faddeev's Green function for the Laplacian.
% The function Htilde can be evaluated using routine Htilde.m.
% 
% Dimension of trigonometric approximation is taken from file ex2DN_comp.m.
% Collection of grid points in the k-plane is created by ex2Kvec_comp.m.
%
% Samuli Siltanen June 2012, Adjusted: Andreas Hauptmann 2017

% Basis functions are cos/sin basis for data from KIT4 
% Order of basis is 15
% load data/ND Ntrig

function Hk=get_Hk_KIT4(k,kkk,K)
% Construct integration points (angles) on the circle
Ntheta = 64;
theta  = 2*pi*[0:(Ntheta-1)]/Ntheta;
theta  = theta(:);
% save data/theta theta Ntheta Dtheta

nBasis  = 15;
Ntrig   = ceil(nBasis/2);
% Loop over points in the k-grid

% load data/ex2Kvec Kvec K1 K2

tMAX = 7;
Kvec = K(abs(K)<tMAX);

if kkk <= length(Kvec)/2
   
    if imag(k)<0            
        Hkfull = precompute_Hktilde_special(k,exp(1i*theta),sqrt(pi),Ntrig);
        A4     = Hkfull(Ntrig+1:end,Ntrig+1:end);
        A3     = Hkfull(1:Ntrig,Ntrig+1:end);
        Accsc   = .5*(A4+flipud(A3));
        Ass_cs  = .5*(A4-flipud(A3));
        Hkfull2 = [real(Accsc),fliplr(imag(Accsc));-flipud(imag(Ass_cs)),rot90(real(Ass_cs),2)];
        Hk      = Hkfull2([1:8,10:end],[1:8,10:end]);
    else
        
        k4     = find(abs(Kvec(1:length(Kvec)/2)-conj(k))<1e-12);
        Hkfull = precompute_Hktilde_special(Kvec(k4),exp(1i*theta),sqrt(pi),Ntrig);
        Hkfull = Hkfull.';
        A4     = Hkfull(Ntrig+1:end,Ntrig+1:end);
        A3     = Hkfull(1:Ntrig,Ntrig+1:end);
        Accsc   = .5*(A4+flipud(A3));
        Ass_cs  = .5*(A4-flipud(A3));
        Hkfull2 = [real(Accsc),fliplr(imag(Accsc));-flipud(imag(Ass_cs)),rot90(real(Ass_cs),2)];
        Hk      = Hkfull2([1:8,10:end],[1:8,10:end]);
    end
else 
    if imag(k)<0
        k4 = find(abs(Kvec(1:length(Kvec))-conj(-k))<1e-12);
        Hkfull = precompute_Hktilde_special(Kvec(k4),exp(1i*theta),sqrt(pi),Ntrig);
        Hkfull = conj(Hkfull);
        A4     = Hkfull(Ntrig+1:end,Ntrig+1:end);
        A3     = Hkfull(1:Ntrig,Ntrig+1:end);
        Accsc   = .5*(A4+flipud(A3));
        Ass_cs  = .5*(A4-flipud(A3));
        Hkfull2 = [real(Accsc),fliplr(imag(Accsc));-flipud(imag(Ass_cs)),rot90(real(Ass_cs),2)];
        Hk      = Hkfull2([1:8,10:end],[1:8,10:end]);
    else
        k4 = find(abs(Kvec(1:length(Kvec))+k)<1e-12);
        Hkfull = precompute_Hktilde_special(Kvec(k4),exp(1i*theta),sqrt(pi),Ntrig);
        Hkfull = Hkfull';
        A4     = Hkfull(Ntrig+1:end,Ntrig+1:end);
        A3     = Hkfull(1:Ntrig,Ntrig+1:end);
        Accsc   = .5*(A4+flipud(A3));
        Ass_cs  = .5*(A4-flipud(A3));
        Hkfull2 = [real(Accsc),fliplr(imag(Accsc));-flipud(imag(Ass_cs)),rot90(real(Ass_cs),2)];
        Hk      = Hkfull2([1:8,10:end],[1:8,10:end]);
    end
                  
end

end
%% precompute operators H_k, 

function Hk = precompute_Hktilde_special(kvec,bz,bl,N)

Nk     = length(kvec);
ds     = abs(bz(2)-bz(1));
Ns     = length(bz);

% points on the boundary (vertical vector),arc-length-parameters
z      = bz(:);
% differences z-y, y also in arc-length-parameters
y      = bz(:).';
ZmY    = repmat(z,1,Ns)-repmat(y,Ns,1);

ci = Ns/2+1;

for iii=1:Nk
%      tic
    k  = kvec(iii);
    % Kernel of the operator, evaluated at z-y
    HK = H1tilde(k*ZmY);
    
    tmp   = conj(fftshift(fft(HK,[],2),2));
    tmpR  = real(tmp(:,[ci-N:ci-1,ci+1:ci+N]));
    tmpI  = imag(tmp(:,[ci-N:ci-1,ci+1:ci+N]));
    tmpR2 = fftshift(fft(tmpR,[],1),1);
    tmpI2 = fftshift(fft(tmpI,[],1),1);    
    Hk  = ds^2/bl*(tmpR2([ci-N:ci-1,ci+1:ci+N],:) + 1i*tmpI2([ci-N:ci-1,ci+1:ci+N],:));
end
end



%% Use the symmetries in Slambda-matrices and compute the matrices in
% quadrants 2,3,4 from the quadrant 1
function [SKq2,SKq3,SKq4] = SK_matrix_compiler(SKq1,kvecq1,kvecq2,kvecq3,kvecq4)

SKq2     = zeros(size(SKq1));
SKq3     = zeros(size(SKq1));
SKq4     = zeros(size(SKq1));
for iii=1:length(kvecq4)
    lambda = kvecq4(iii);
    Sl     = SKq1(:,:,abs(kvecq1-conj(lambda))<1e-10);
    SKq4(:,:,iii) = Sl.';
end
for iii=1:length(kvecq3)
    lambda = kvecq3(iii);
    Sl     = SKq1(:,:,abs(kvecq1+lambda)<1e-10);
    SKq3(:,:,iii) = Sl';
end
for iii=1:length(kvecq2)
    lambda = kvecq2(iii);
    Sl     = SKq1(:,:,abs(kvecq1+conj(lambda))<1e-10);
    SKq2(:,:,iii) = conj(Sl);
end
end
