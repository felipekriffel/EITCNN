% Computation of the Dirichlet-to-Neumann map as a matrix acting 
% in trigonometric basis. Omega is the unit disc.
%
% We actually precompute the Neumann-to-Dirichlet map using FEM,
% and then find DN map by inverting the ND map and adding 
% appropriate mapping of constant functions.
%
% We compute in addition the DN map related to the constant 
% conductivity 1 (analytically)
% and save both DN matrices to file data/DN.mat.
% 
% Routine ND_comp.m must be run before this file.
%
% Samuli Siltanen April 2014. modified: Andreas Hauptmann 2017

% Load precomputed data
load data/ND NtoD NtoD1
% Invert the ND matrix to get DN matrix in trigonometric basis apart
% from constant functions.
DN = inv(NtoD);
DN1 = inv(NtoD1);

parVal=ceil(L/2);
% % Add appropriate zero row and zero column

DN = [DN(:,1:parVal),DN(:,parVal+1:end)];
DN = [DN(1:parVal,:);DN(parVal+1:end,:)];

DN1 = [DN1(:,1:parVal),DN1(:,parVal+1:end)];
DN1 = [DN1(1:parVal,:);DN1(parVal+1:end,:)];

% Save result to file
save data/DN DN DN1
