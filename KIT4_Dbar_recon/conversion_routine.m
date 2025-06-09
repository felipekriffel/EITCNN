SAVEPATH = '/home/feliperiffel/Documentos/Mestrado/EITCNN/dbar_data/dbar_recons/first_exp/dbar_img/'; 
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
    
end