arg_list = argv();

% DIRPATH = '/home/feliperiffel/EITCNN_git/dbar_data/data_matrix'; % substitua pelo caminho real
% SAVEPATH = '/mnt/c/Users/Felipe/Documents/dbar_results/first_exp/dbar_img/';

if nargin != 2
    error("Not enough arguments \n **Usage**: octave recon_routine.m path/to/data_directory path/to/save_directory \n");
end

DIRPATH = arg_list{1}; % substitua pelo caminho real
SAVEPATH = arg_list{2};

files = dir(fullfile(DIRPATH, '*.mat')); % você pode trocar a extensão se necessário

if length(files)==0
    printf("\n *** Warning: no .mat files in data directory *** \n");
end

load data/tri_index

x = linspace(-1,1,128);
[mx,my] = meshgrid(x,-x);

plotFlag=false;
printFlag=true;

for k = 1:(length(files))
    filename = files(k).name;
    filepath = fullfile(DIRPATH, filename);
    
    % Carrega o arquivo .mat
    fprintf('Carregando: %s\n', filename);
    dados = load(filepath);
    %% Evaluation of data
    %Load measured data
    %eval(['load KIT4_measdata/dataMat_adj_' num2str(ex) '_' num2str(ver)]);
    eval(['load ' filepath]);

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

    dbar_img = avalia_ef_tri(p,t,recon,mx,my,tri_index);

    save('-mat7-binary',fullfile(SAVEPATH, strrep(filename,"_input.mat",".mat")),'dbar_img');

end