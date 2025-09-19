% This is the main script for reading data from the KIT4 system and
% applying the D-bar method to obtain a reconstruction for all examples.
%
% written by Andreas Hauptmann, 2017


% Choose experiment number in 1-8
ex=2;
% Choose data set in experiment, the maximum number of individual 
% measurements for each experiment is: [4 6 6 4 2 7 2 6]
ver=0;

plotFlag=true;
printFlag=true;

%% Evaluation of data
%Load measured data
%eval(['load KIT4_measdata/dataMat_adj_' num2str(ex) '_' num2str(ver)]);
eval(['load dbar_data/data_matrix/sample_' num2str(ex) '_' num2str(ver) '.mat']);

%Save for evaluation in change of basis
save data/KIT4_measurement U_ad10 U_ad0
        
      

%Transform adjacent measurements to ND map (separately)
comp02_ND_buildFromKIT4
%Convert to DN map
comp03_DN_build
%Solve BIE for CGO solutions
comp04_psi_BIE
%Compute scattering transform
comp05_tBIE_psi   

%Solve D-bar equation
comp06_Dbarsolve


plotFlag=true
printFlag = true

% plot results
if(plotFlag)
    load data/reconstruction p e t recon
    figure(2)
    clf
    x = p(1, :)';
    y = p(2, :)';
    tri = t(1:3, :)';  % conectividade dos triângulos
    u = recon(:);      % garante formato coluna
    
    figure('Position', [100, 100, 800, 700]);
    trisurf(tri, x, y, u);
    shading interp;
    colormap jet;
    colorbar;
    view(2);           % vista 2D, como em pdeplot
    xlabel('x'); ylabel('y');      
%     colorbar off, 
    axis equal;
    axis padded;  % Ou axis image
end        
if(printFlag)
    if ~exist('KIT4_recons', 'dir')
        mkdir('KIT4_recons');
    end
    %filename = ['KIT4_recons/phantom_' num2str(ex) '_' num2str(ver) '.jpeg'];
    filename = ['gendata_recons/phantom_' num2str(ex) '_' num2str(ver) '.jpeg'];
    disp(filename)
    print(filename, '-djpeg');
end

