# UNET_train_npy.py
import os
import glob
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from unet import UNetCompiled  # Certifique-se de que esse arquivo está no mesmo diretório

def create_sample_dataset(npy_folder, batch_size, epochs):
    files = sorted(glob.glob(os.path.join(npy_folder, "*.npy")))

    def generator():
        for file in files:
            data = np.load(file).astype(np.float32)  # (2, 128, 128)
            entrada = data[0]  # imagem D-bar
            saida = data[1]    # imagem esperada
            yield entrada, saida

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(128, 128), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 128), dtype=tf.float32),
        )
    )
    dataset = dataset.map(lambda x, y: (tf.expand_dims(x, -1), tf.expand_dims(y, -1)))  # (128, 128, 1)
    dataset = dataset.repeat(epochs).batch(batch_size).prefetch(1)
    return dataset

def main(SETTINGS_JSON):
    if SETTINGS_JSON.endswith('.json') and os.path.isfile(SETTINGS_JSON):
        with open(SETTINGS_JSON) as f:
            settings = json.loads(f.read())
    else:
        settings = json.loads(SETTINGS_JSON)

    savepath = settings["save_dir"]
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    batch_size = settings["batch_size"]
    epochs = settings["epochs"]
    dropout = settings["dropout_prob"]
    train_path = settings["train_path"]
    val_path = settings["val_path"]
    steps_per_epoch = settings["steps_per_epoch"]
    save_period = settings["save_period"]

    dataset = create_sample_dataset(train_path, batch_size, epochs)
    dataset_val = create_sample_dataset(val_path, batch_size, epochs)

    model = UNetCompiled(input_size=(128, 128, 1), n_filters=32, n_classes=1, dropout=dropout)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='mean_squared_error',
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(savepath, "checkpoints", "epoch_{epoch:02d}.keras"),
        save_weights_only=False,
        save_best_only=False,
        save_freq=(len(glob.glob(train_path + "/*.npy")) // batch_size) * save_period,
    )

    history = model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=dataset_val,
        verbose=1,
        callbacks=[checkpoint]
    )

    # Salvar modelo
    model.save(os.path.join(savepath, 'unet_final.keras'))

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(savepath, "training_graph.png"))
    plt.close()

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
