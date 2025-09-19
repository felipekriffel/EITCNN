%% This routine will create the necessary indices and matrices for
% the computation of the scattering transform. Because of symmetries in the
% S-lambda matrices we only need quadrant1 (q1) (or q1,q2) for their computation, and
% the other quadrants will use those modified matrices, this is why this
% routine will output objects for every quadrant.
function [kmat,indq1,indq2,indq3,indq4,kvecq1,kvecq2,kvecq3,kvecq4] = k_grid_creator(kmax,Nk)

[k1,k2]   = meshgrid(linspace(-kmax,kmax,Nk));
kmat      = k1+1i*k2;
indq1     = (real(kmat)>0) & (imag(kmat)>0);
indq2     = (real(kmat)<0) & (imag(kmat)>0);
indq3     = (real(kmat)<0) & (imag(kmat)<0);
indq4     = (real(kmat)>0) & (imag(kmat)<0);
kvecq1    = kmat(indq1);
kvecq2    = kmat(indq2);
kvecq3    = kmat(indq3);
kvecq4    = kmat(indq4);
end