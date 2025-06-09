SAVEPATH = '/home/feliperiffel/Documentos/Mestrado/EITCNN/dbar_data/dbar_recons/'; 
DIRPATH = 'gendata_recons';
files = dir(fullfile(DIRPATH, '*.mat')); 

load data/tri_index

% Itera sobre cada arquivo
for k = randperm(length(files))
    filename = files(k).name;
    filepath = fullfile(DIRPATH, filename);
    
    % Carrega o arquivo .mat
    fprintf('Carregando: %s\n', filename);
    dados = load(filepath);
    eval(['load ' filepath]);

    save('-mat7-binary',[SAVEPATH filename],'dbar_img');

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