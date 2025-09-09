import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from scipy.ndimage import rotate
import tensorflow as tf


def load_dbar_data(folder_path):
    """
    Carrega os arquivos .npy com dados simulados do Deep-Dbar.
    Cada arquivo deve ter formato (2, 128, 128) → canais: [entrada, saída esperada]
    """
    x_data = []
    y_data = []
    filenames = sorted([f for f in os.listdir(folder_path) if f.endswith('.npy')])

    for file in filenames:
        full_path = os.path.join(folder_path, file)
        data = np.load(full_path)  # (2, 128, 128)
        data = np.transpose(data, (1, 2, 0))  # (128, 128, 2)

        x = np.expand_dims(data[..., 0], axis=-1)  # (128, 128, 1)
        y = data[..., 1]  # (128, 128)

        x_data.append(x)
        y_data.append(y)

    return np.array(x_data), np.array(y_data), filenames


def predict_and_save(x_data, y_data, model_path, results_path, filenames):
    """
    Carrega o modelo, realiza predições e salva imagens comparando predição com ground truth.
    """
    os.makedirs(results_path, exist_ok=True)

    model = keras.models.load_model(model_path)
    predictions = model.predict(x_data)

    for i in range(len(predictions)):
        pred = rotate(predictions[i].squeeze(), 180)
        true = y_data[i]

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(pred, cmap='plasma', vmin=0, vmax=1)
        axs[0].set_title("Predição (UNet)")
        axs[0].axis('off')

        axs[1].imshow(true, cmap='plasma', vmin=0, vmax=1)
        axs[1].set_title("Ground Truth")
        axs[1].axis('off')

        plt.tight_layout()
        out_file = os.path.join(results_path, f"predict_{filenames[i].replace('.npy', '.png')}")
        plt.savefig(out_file)
        plt.close()


def main():
    test_folder = "dbar_data-new/test"
    model_file = "EIT_model_npy/unet.keras"
    output_folder = "results_unet"

    x_data, y_data, filenames = load_dbar_data(test_folder)
    predict_and_save(x_data, y_data, model_file, output_folder, filenames)


if __name__ == "__main__":
    main()
