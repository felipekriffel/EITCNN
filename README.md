# CEM_CNN

Testes de redes neurais na reconstrução de imagens de Tomografia por Impedância.

## Ramo `main`

Arquivos para implementação dos testes com o método Deep-DSM.


### `CNN_routine.py`

Faz a rotina de geração, treino e teste de acordo com os arquivos de configuração informados.

### `DSM_test_kit4.py'`

Carrega uma rede treinada e testa nas amostras do conjunto de dados KIT4. 

Argumentos: 
- `results_path`: caminho para pasta com a rede treinada e onde o resultado do teste será salvo.


### `CEM_data_gen.py`

Gera dados aleatórios de condutividade, salvando os vetores de coeficientes e as informações usadas para gerar cada imagem.

### `DSM_data_gen`

Gera dados com as diferenças de Cuachy para o método DSM, usando como base as condutividades previamente geradas.

### `CNN_test_checkpoints`

Usado para treinar a UNET

### `eitx.py`

Arquivo com as funções usadas na implementação da Tomografia por Impedância (Modelo Completo de Eletrodos)

### `UNET_train.py`

Treinamento da rede UNET definida.

### `unet.py`

Arquivo com a classe da rede UNET usada.