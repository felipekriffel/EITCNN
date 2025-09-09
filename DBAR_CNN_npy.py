import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dolfinx
import pyvista
import eitx
import os
import json 
import sys

def carregar_dados_de_pasta(pasta):
    arquivos = sorted([f for f in os.listdir(pasta) if f.endswith(".npy")])
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
    model = tf.keras.models.load_model(model_path)
    print("✅ Modelo carregado")
    predictions = model.predict(input_val)
    print(f"✅ Previsões geradas: {predictions.shape}")
    for i in range(min(10, predictions.shape[0])):
        plt.imshow(predictions[i].squeeze(), cmap='viridis')
        plt.colorbar()
        plt.title(f"Predição {i}")
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, f"predicao_{i}.png"))
        plt.close()
    print(f"✅ Imagens salvas em: {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Uso: python predict_unet_from_npy.py caminho_para_modelo caminho_para_dados_npy")
    else:
        model_path = sys.argv[1]
        data_path = sys.argv[2]
        main(model_path, data_path)
