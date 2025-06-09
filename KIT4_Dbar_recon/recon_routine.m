DIRPATH = '/home/feliperiffel/Documentos/Mestrado/EITCNN/dbar_data/data_matrix'; % substitua pelo caminho real
SAVEPATH = 'gendata_recons';
files = dir(fullfile(DIRPATH, '*.mat')); % você pode trocar a extensão se necessário

load data/tri_index

x = linspace(-1,1,128);
[mx,my] = meshgrid(x,-x);

plotFlag=false;
printFlag=true;

% Itera sobre cada arquivo
for k = randperm(length(files))
    filename = files(k).name;
    filepath = fullfile(DIRPATH, filename);
    
    % Carrega o arquivo .mat
    fprintf('Carregando: %s\n', filename);
    dados = load(filepath);
    
    % Aqui você pode fazer algo com 'dados'
    % Por exemplo, listar as variáveis carregadas:
    % disp(fieldnames(dados));

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

    save(['gendata_recons/' strrep(filename,".mat","_dbar.mat")],'dbar_img');

    % % plot results
    % if(plotFlag)
    %     load data/reconstruction p e t recon
    %     figure(2)
    %     clf
    %     x = p(1, :)';
    %     y = p(2, :)';
    %     tri = t(1:3, :)';  % conectividade dos triângulos
    %     u = recon(:);      % garante formato coluna
        
    %     figure('Position', [100, 100, 800, 700]);
    %     trisurf(tri, x, y, u);
    %     shading interp;
    %     colormap jet;
    %     colorbar;
    %     view(2);           % vista 2D, como em pdeplot
    %     xlabel('x'); ylabel('y');      
    % %     colorbar off, 
    %     axis equal;
    %     axis padded;  % Ou axis image
    % end        
    % if(printFlag)
    %     if ~exist('KIT4_recons', 'dir')
    %         mkdir('KIT4_recons');
    %     end
    %     %filename = ['KIT4_recons/phantom_' num2str(ex) '_' num2str(ver) '.jpeg'];
    %     filename = ['gendata_recons/phantom_' num2str(ex) '_' num2str(ver) '.jpeg'];
    %     disp(filename)
    %     print(filename, '-djpeg');
    % end
    
end