% Approximate reconstruction of conductivity from truncated scattering data
% using the D-bar method.
%
% We optimize the computation by restricting the degrees of freedom to the
% values of the solution at grid points satisfying |k|<R.
%
% Samuli Siltanen March 2013, slight adjustments: Andreas Hauptmann 2017


load data/mesh
xvec = p(1,:);
yvec = p(2,:);
zvec = xvec(:)+1i*yvec(:);
Nz  = length(zvec);

% Load precomputed scattering transform and its evaluation points
load data/ex2Kvec Kvec K1 K2 tt tMAX
load data/ex2tBIE tBIE


scatBIE_34 = zeros(size(K1));
scatBIE_34(abs(K1+1i*K2)<tMAX) = tBIE;

%Additional cut off for max values (needed for partial data)
cOff=25;
scatBIE_34(abs(real(scatBIE_34))>cOff)=0;
scatBIE_34(abs(imag(scatBIE_34))>cOff)=0;

scatBIE=scatBIE_34;

% Choose truncation radius R>0  
R=4;
% Choose parameter M for the computational grid
M = 6; %8 
N = 2^M;

if R>tMAX
    error(['N_recon.m: Truncation radius R=', num2str(R), ' too big, must be less than ', num2str(tMAX)])
end

% Construct grid points
[k1,k2,h,tmp,tmp,tmp] = GV_grids(M, M+1, 2.3*R);
k    = k1 + 1i*k2;
Rind = abs(k)<R;
Nind = round(sum(sum(double(Rind))));

% Evaluate scattering transform at the grid points using precomputed values
% and two-dimensional interpolation
scatvec = zeros(Nind,1);
k1vec   = k1(Rind);
k2vec   = k2(Rind);

%again in parallel
for iii = 1:Nind
    if mod(iii,100)==0
        disp([iii Nind])
    end
    scatvec(iii) = interp2(K1,K2,scatBIE,k1vec(iii),k2vec(iii),'cubic');
end
scat = zeros(size(k1));
scat(Rind) = scatvec;

max(scat(:))
%%
% Evaluate scattering transform divided by conj(k). Avoid singularity
% at the origin by setting value there to zero.
ktmp        = k;
ind0        = (abs(k)<1e-14); % Location of the origin in the grid
ktmp(ind0)  = 1;
scatk       = scat./conj(ktmp);
scatk(ind0) = 0;

% Evaluate Green's function 1/(pi*k). Avoid singularity at the origin by
% setting value there to zero.
fund       = 1./(pi*ktmp);
fund(ind0) = 0;

% Smooth truncation of Green's function near the boundary
s  = abs(min(min(k1)));
ep = s/10;
RR = (s-ep)/2;
bigind       = abs(k)>=s;
fund(bigind) = 0;
medind       = (abs(k)<s) & (abs(k)>2*RR);
fund(medind) = fund(medind).*(1-(abs(k(medind))-2*RR)/ep);

% Take FFT of the fundamental solution at this time
fundfft    = fft2(fftshift(fund));

% Construct right hand side of the Dbar equation
rhs = [ones(Nind,1);zeros(Nind,1)];

% Initialize reconstruction
recon = ones(Nz,1);
iniguess = [ones(Nind,1);zeros(Nind,1)];

% Loop over points of reconstruction (in parallel)
for iii = 1:Nz
  
    
    % Current point of reconstruction
    z = zvec(iii);
    
    % Construct multiplicator function for the Dbar equation
    TR = 1/(4*pi)*scatk.*(exp(-i*(k*z+conj(k*z))));
    
    % Solve the real-linear D-bar equation with gmres keeping the real and
    % imaginary parts of the solution separate
    [w,tmp,tmp,tmp,tmp] = gmres('DB_oper', rhs, 50, 1e-5, 500, [], [], iniguess, fundfft, TR, k1, k2, M, h, k, R, Rind, Nind);
    
    % Use the current solution as the next initial guess
%     iniguess = w;
    
    % Construct solution mu inside the unit disc
    mu = zeros(size(k1));
    mu(Rind) = w(1:Nind) + i*w((Nind+1):end);
    
    % Pick out the reconstructed conductivity value
    recon(iii) = (mu(ind0))^2;
    
    % Monitor the run
    if mod(iii,100)==0
        disp([iii Nz])
    end
end

% Save results to file
recon=real(recon);
save data/reconstruction p e t recon


