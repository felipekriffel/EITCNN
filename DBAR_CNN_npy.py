import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import json 
import sys
import numpy as np
import scipy
from PIL import Image
from scipy.ndimage import rotate
EXPERIMENTOS = ['1_1','1_3','1_4','2_2','2_3','2_4','2_5','2_6',
                '3_1','3_4','3_2','3_6','4_1','4_3','4_4','5_2']

def carregar_dados_de_pasta(pasta):
    

    arquivos = sorted([f for f in os.listdir(pasta) if f.endswith(".npy")])
    arquivos = [f for f in arquivos if f[-12:-9] in EXPERIMENTOS]
    print(arquivos)
    dados = []
    for nome in arquivos:
        caminho = os.path.join(pasta, nome)
        amostra = np.load(caminho)
        x_img = amostra[0]
        dados.append(np.transpose(x_img))
    dados_array = np.array(dados)
    dados_array = np.expand_dims(dados_array, axis=-1)
    return tf.convert_to_tensor(dados_array, dtype=tf.float32)

def main(model_path, data_path, output_dir="resultados_unet"):
    os.makedirs(output_dir, exist_ok=True)
    input_val = carregar_dados_de_pasta(data_path)
    print(f"✅ Dados carregados: {input_val.shape}")
    model = tf.keras.models.load_model(model_path,compile=False)
    model.compile()
    

    result = model.predict(input_val)

    photo_array = []
    for exper in EXPERIMENTOS:
        path = os.path.join("fin_data", "target_photos", f"fantom_{exper}.jpg")
        img = np.asarray(Image.open(path))
        photo_array.append(img)
    
    print(len(photo_array))
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(result.shape[0], 2, figsize=(10, 40))
    img_array = []
    for k in range(result.shape[0]):
        img_array.append(ax[k][0].imshow(result[k], cmap='viridis', vmin=-1.0, vmax=1.0))
        ax[k][0].set_axis_off()
        ax[k][1].imshow(photo_array[k])
        ax[k][1].set_axis_off()
    fig.colorbar(img_array[0], ax=ax, orientation='vertical')
    plt.savefig(os.path.join("dbar_test_result.png"))
    
    print("✅ Resultado salvo em:", os.path.join("dbar_test_result.png"))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python predict_unet_from_npy.py caminho_para_modelo caminho_para_dados_npy")
    else:
        model_path = sys.argv[1]
        data_path = sys.argv[2]
        main(model_path, data_path)
