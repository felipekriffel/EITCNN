# CEM_CNN

Testes de redes neurais na reconstrução de imagens de Tomografia por Impedância, em particular do método Deep - DSM. Nesse processo, construímos redes neurais que recebem como entrada a solução de uma EDP auxiliar, retornando uma imagem com a segmentação das inclusões na condutividade procurada. Para mais detalhes:

- Artigo original (GUO & JIANG, 2020): https://arxiv.org/abs/2009.08024
- Minha dissertação de mestrado sobre o tema: https://pergamum.ufsc.br/acervo/398757

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

Gera dados com as diferenças de Cauchy para o método DSM, usando como base as condutividades previamente geradas.

### `eitx.py`

Arquivo com as funções usadas na implementação da Tomografia por Impedância (Modelo Completo de Eletrodos)

### `UNET_train.py`

Treinamento da rede UNET definida.

### `unet.py`

Arquivo com a classe da rede UNET usada.
