import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate
import tensorflow as tf
import eitx
import dolfinx

# ===== CONFIGURAÇÕES =====
DATAMAT_PATH = "fin_data/datamat"          # pasta com datamat_*.mat
FILEPATH = "./"                        # pasta raiz (para 'fin_data/')
MODELPATH = "EIT_model_npy/"                 # subpasta com modelo unet.keras
RESULTS_PATH = "resultados_unet/"      # onde salvar os resultados
EXPERIMENTOS = ['1_1','1_2','1_3','1_4','2_2','2_3','2_4','2_5','2_6',
                '3_1','3_2','3_6','3_4','3_5','4_1','4_3','4_4','5_2']
currents = list(range(15))             # índices dos experimentos

# ===== PRÉ-PROCESSAMENTO =====
def processar_dados_experimentais():
    # Carrega dados de fundo
    mat = scipy.io.loadmat(os.path.join(DATAMAT_PATH, "datamat_1_0"))
    Uel = mat["Uel"].T
    CP = mat["CurrentPattern"].T
    Uel_b = Uel[-15:][currents]

    # Converte dados de fundo
    list_U0_m = np.array([eitx.ConvertingData(u, method="KIT4") for u in Uel_b])
    list_U0 = list_U0_m.flatten()
    I_all = CP[-15:][currents] / np.sqrt(2)
    l, L = I_all.shape

    # Eletrodos e malha
    ele_pos = eitx.Electrodes(L, 0.5, 0)
    mesh_object = eitx.MeshClass(ele_pos, 0.4, 0.6)
    dir_problem = eitx.DirectProblem(mesh_object, np.ones(L)*0.07858)

    # Domínio homogêneo
    radius = 1
    N = 128
    h = 2 * radius / (N - 1)
    x = [radius - i * h for i in range(N)]
    y = [-radius + i * h for i in range(N)]
    mesh_x, mesh_y = np.meshgrid(x, y)

    # Background homogêneo
    gamma0 = dolfinx.fem.Function(dir_problem.V0)
    gamma0.x.array[:] = 1.2

    # Preparar entradas
    T1 = []
    for exper in EXPERIMENTOS:
        mat = scipy.io.loadmat(os.path.join(DATAMAT_PATH, f"datamat_{exper}"))
        Uel_f = mat["Uel"].T[-15:][currents]
        list_U1_m = np.array([eitx.ConvertingData(u, method="KIT4") for u in Uel_f])
        differ = [list_U1_m[k] - list_U0_m[k] for k in range(len(list_U0_m))]
        list_ur_dif, _ = dir_problem.solve_problem_current(differ, gamma0)

        T = np.zeros((l + 2, N, N))
        for k in range(l):
            T[k] = eitx.genPotentialImg(list_ur_dif[k], mesh_x, mesh_y, 1.2)
        T[l] = mesh_x
        T[l+1] = mesh_y
        T1.append(np.transpose(T))  # shape: (128, 128, canais)

    input_val = tf.convert_to_tensor(T1, dtype=tf.float32)
    input_val = tf.expand_dims(input_val, axis=-1)  # shape: (n_exper, 128, 128, 1)
    return input_val, x, y

# ===== PREDIÇÃO =====
def prever_com_modelo(input_val, x, y):
    model = tf.keras.models.load_model(os.path.join(FILEPATH, MODELPATH, "unet_final.keras"),compile=False)
    model.compile()
    print("✅ Modelo carregado")
    classes = model.predict(input_val)
    
    n_exper, N, _, _ = classes.shape
    result = 0.5 * np.ones((n_exper, N, N))
    for k in range(n_exper):
        rotated = rotate(classes[k], 180)
        for i in range(N):
            for j in range(N):
                if x[i]**2 + y[j]**2 > 1**2:
                    rotated[i][j] = 0.5
        result[k, :, :] = rotated[:, :, 0]
    return result

# ===== PLOTAGEM =====
def plotar_resultados(result):
    photo_array = []
    for exper in EXPERIMENTOS:
        path = os.path.join(FILEPATH, "fin_data", "target_photos", f"fantom_{exper}.jpg")
        img = np.asarray(Image.open(path))
        photo_array.append(img)

    os.makedirs(RESULTS_PATH, exist_ok=True)
    fig, ax = plt.subplots(result.shape[0], 2, figsize=(10, 40))
    img_array = []
    for k in range(result.shape[0]):
        img_array.append(ax[k][0].imshow(result[k], cmap='viridis', vmin=-1.0, vmax=1.0))
        ax[k][0].set_axis_off()
        ax[k][1].imshow(photo_array[k])
        ax[k][1].set_axis_off()
    fig.colorbar(img_array[0], ax=ax, orientation='vertical')
    plt.savefig(os.path.join(RESULTS_PATH, "test_result.png"))
    print("✅ Resultado salvo em:", os.path.join(RESULTS_PATH, "test_result.png"))

# ===== EXECUÇÃO =====
if __name__ == "__main__":
    input_val, x, y = processar_dados_experimentais()
    print("✅ Dados experimentais preparados:", input_val.shape)
    result = prever_com_modelo(input_val, x, y)
    print("✅ Predição finalizada:", result.shape)
    plotar_resultados(result)
