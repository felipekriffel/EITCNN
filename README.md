# CEM_CNN

Testes de redes neurais na reconstrução de imagens de Tomografia por Impedância.

## Ramo `dbar`

Códigos para implementação do Deep dbar. Nele, cálculamos uma imagem de EIT via método DBAR e informamos como entrada numa rede convolucional para melhorá-la. Referência: https://ieeexplore.ieee.org/document/8352045.

Os códigos foram separados e organizados para rodar em experimentos. Salvamos as informações de cada experimento em arquivos `exp_settings.json`. Há dois tipos de configurações: `data_gen_settings` e `dbar_settings`. O primeiro, para geração dos dados de condutividade. O segundo, para geração dos dados de DBAR.

- `data_gen_settings`:
    - `"datapath":` `String`, diretório de armazenamento dos dados gerados
    - `"L":` `int`, número de eletrodos
    - `"bg":` `float`, valor da condutividade no background
    - `"ivhigh":` `float`,  valor da condutividade nas inclusões condutivas 
    - `"ivlow":` `float`,  valor da condutividade nas inclusões resistivas
    - `"p_ivhigh"` `float`, probabilidade de selecionar uma inclusão condutiva(valor entre 0 e 1)
    - `"p_ivlow":` `float`, probabilidade de selecionar uma inclusão resistiva
    - `"n_samples":` `list`,  número de amostras para cada número de inclusões. Exemplo `[4,6,1]` 4 amostras c/ 1 inclusão, 6 c/ 2 inclusões, etc.
    - `"noise_level":` `float` nível de ruído relativo nos dados.


`"samples_dir"`: `String`,, diretório com as amostras
`"dbar_input_datapath"`: `String`,, diretório de armazenamento dos dados de potenciais e imagens de condutividade gerados
`"dbar_mat_datapath"`String,, diretório de armazenamento dos 
`"dbar_img_datapath"`String,, diretório de armazenamento das amostras juntando imagens dbar e de classificação alvo
`"currents"`: `list`,,  conjunto de correntes usadas
`"L"`: `int`, , número de eletrodos
`"bg"`: `float`, , valor da condutividade no background
`"ivhigh"`: `float`, ,  valor da condutividade nas inclusões condutivas 
`"ivlow"`: `float`, ,  valor da condutividade nas inclusões resistivas
`"p_ivhigh"` float, , probabilidade de selecionar uma inclusão condutiva(valor entre 0 e 1)
`"p_ivlow"`: `float`, , probabilidade de selecionar uma inclusão resistiva
`"n_samples"`: `list`, ,  número de amostras para cada número de inclusões. Exemplo [4,6,1] 4 amostras c/ 1 inclusão, 6 c/ 2 inclusões, etc. 
`"noise_level"`: `float`, , nível de ruído relativo nos dados
`"N"`: `int` , resolução N x N das imagens usadas na classificação


### `CEM_data_gen.py`

Gera dados aleatórios de condutividade, salvando os vetores de coeficientes e as informações usadas para gerar cada imagem.

Uso: `python3 CEM_data_gen.py path/to/settings.json` ou `python3 CEM_data_gen.py {dumped_json_file}`

### `DBAR_data_gen.py`

Gera dados de potenciais, matrizes de referência para entrada no dbar e as imagens alvo para classificação na rede neural, usando como base as condutividades previamente geradas.

Uso: `python3 DBAR_data_gen.py path/to/settings.json` ou `python3 DBAR_data_gen.py {dumped_json_file}`

### `DBAR_dataprep.py`

Junta as saídas do dbar em formato `.mat`, retornadas pelo método implementado em `octave`, com as imagens alvo de condutividade em `.npy`, geradas pelo código `DBAR_data_gen.py`.

Uso: `python3 DBAR_dataprep.py path/to/settings.json` ou `python3 DBAR_dataprep.py {dumped_json_file}`

### `DBAR_unet_train.py`

Treinamento da rede UNET definida para os dados do dbar.

Uso:`python3 DBAR_unet_train.py path/to/settings.json` ou `python3 DBAR_unet_train.py {dumped_json_file}`

### `dbar_octave/`

Arquivos em `octave` para execução do método dbar, adaptados do código `MATLAB` desenvolvido em https://fips.fi/blog/the-d-bar-method-for-electrical-impedance-tomography-experimental-data/.

#### `recon_routine.m`

Executa o método dbar para uma série de amostras presentes no diretório informado, salvando em outro destino também informado

`octave recon_routine.m path/to/data_directory path/to/save_directory`

Cada amostra em `path/to/data_directory` deve ser um arquivo `.mat` com as entradas:
- `"U_ad0"`: matriz de potenciais para a condutividade que se deseja reconstruir;
- `"U_ad10"`: matriz de potenciais para a condutividade somente com o background
- `"MeasPat"`: padrão de potenciais medidos. 