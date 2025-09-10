import os
import shutil
import random
import numpy as np

def dividir_e_normalizar_dados(pasta_base, proporcao_val=0.2, normalizacao='minmax', seed=42, sobrescrever=True):
    random.seed(seed)

    pasta_train = os.path.join(pasta_base, "train")
    pasta_val = os.path.join(pasta_base, "val")

    # Criação das pastas
    if sobrescrever or not (os.path.exists(pasta_train) and os.path.exists(pasta_val)):
        os.makedirs(pasta_train, exist_ok=True)
        os.makedirs(pasta_val, exist_ok=True)

        arquivos = [f for f in os.listdir(pasta_base)
                    if f.endswith(".npy") and os.path.isfile(os.path.join(pasta_base, f))]
        random.shuffle(arquivos)

        n_val = int(len(arquivos) * proporcao_val)
        val_files = arquivos[:n_val]
        train_files = arquivos[n_val:]

        for f in train_files:
            shutil.move(os.path.join(pasta_base, f), os.path.join(pasta_train, f))
        for f in val_files:
            shutil.move(os.path.join(pasta_base, f), os.path.join(pasta_val, f))

        print(f"✅ {len(train_files)} arquivos movidos para: {pasta_train}")
        print(f"✅ {len(val_files)} arquivos movidos para: {pasta_val}")
    else:
        print("🔄 Subpastas já existem e foram mantidas.")
    return n_val
