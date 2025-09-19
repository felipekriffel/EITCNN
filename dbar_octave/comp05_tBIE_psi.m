% This routine is for computing the scattering transform at the k-grid
% created by routine ex2Kvec_comp.m (and saved to file data/ex2Kvec.mat).
%
% We load from file the Fourier coefficients of the traces of Faddeev 
% exponentially growing solutions created by routine ex2psi_BIE_comp.m 
% (and saved to file data/ex2psi_BIE.mat).
% The traces are evaluated at the following points on the unit circle:
% exp(i\theta), where vector \theta is previously saved to file data/theta.mat.
%
% Then we integrate according to the definition 
%
%  tBIE(k) = \int_{boundary} e^{i*conj(kx)} (Lg-L1) psi(x,k) d\sigma(x)
%
% using a loop over all k values in the vector Kvec.
% The result is saved to file data/ex2tBIE.mat.
%
% Samuli Siltanen June 2012, adjusted to KIT4 data: Andreas Hauptmann 2017

% Load vectors of k and theta values
load data/ex2Kvec Kvec
load data/theta theta Ntheta Dtheta
load data/psi_BIE Fpsi_BIE 

% Initialize the result
tBIE = zeros(size(Kvec));
% Load precomputed DN maps
load data/DN DN DN1

%order of cosine/sine basis
Ntrig=[1:8 7:-1:1];
%We have only 15 independent pattern
nBasis=15;

%Compute in parallel (change to for if parallel computing tool box not
%available)
for iii = 1:length(Kvec)
    k = Kvec(iii);
    
    cald  = (exp(1i*k*exp(1i*theta))); 
    
%     Fpsi=ones(length(Ntrig),1);

%     for jjj = 1:length(Ntrig)
%         if jjj<=(nBasis+1)/2
%         Fpsi(jjj) = 1/sqrt(pi)*Dtheta*(cos(Ntrig(jjj)*theta))'*cald;
%         else
%         Fpsi(jjj) = 1/sqrt(pi)*Dtheta*(sin(Ntrig(jjj)*theta))'*cald;
%         end
%     end
    

    
    % Apply DN maps to evaluate (Lg-L1) psi(x,k)
    FLLpsi = (DN-DN1)*Fpsi_BIE(:,iii);
    LLpsi  = zeros(size(theta));

    for jjj = 1:length(Ntrig)
        if jjj<=(nBasis+1)/2
            LLpsi = LLpsi + 1/sqrt(pi)*FLLpsi(jjj)*(cos(Ntrig(jjj)*theta));
        else
            LLpsi = LLpsi + 1/sqrt(pi)*FLLpsi(jjj)*(sin(Ntrig(jjj)*theta));
        end
    end
    
    % Integrate 
    tBIE(iii) = Dtheta*exp(1i*conj(k)*exp(-1i*theta.'))*LLpsi;
    
    %Monitor the run
    if mod(iii,100)==0
        disp(['Done ', num2str(iii), ' out of ', num2str(length(Kvec))])
    end
end

% Save the result to file.
load data/ex2Kvec Kvec tt tMAX
save data/ex2tBIE tBIE Kvec tt tMAX

