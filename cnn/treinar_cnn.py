"""
=====================================================================
 TREINAMENTO E AVALIAÇÃO DA CNN
=====================================================================
Uso (a partir da pasta do projeto):
    python cnn/treinar_cnn.py

Validação: Hold-out estratificado 70% treino / 30% teste (mesmo split da MLP).
Dentro dos 70% de treino, 15% são separados como validação apenas para o
Early Stopping — o conjunto de TESTE só é usado no final, para as métricas.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import keras
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from comum import config
from comum.dados import dividir_holdout, abrir_imagem
from comum.metricas import avaliar
from cnn.modelo_cnn import criar_modelo_cnn, preparar_imagem

BATCH_SIZE = 32
EPOCHS = 30


def carregar_pixels(caminhos):
    return np.stack([preparar_imagem(abrir_imagem(p)) for p in caminhos]).astype("float32")


if __name__ == "__main__":
    keras.utils.set_random_seed(config.SEMENTE)

    # 1. Mesma divisão hold-out usada pela MLP
    cam_treino, y_treino, cam_teste, y_teste, classes = dividir_holdout()
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Treino: {len(cam_treino)} imagens | Teste: {len(cam_teste)} imagens")

    print("Carregando pixels das imagens (224x224)...")
    X_treino_total = carregar_pixels(cam_treino)
    X_teste = carregar_pixels(cam_teste)

    # 2. Validação interna (só para o early stopping), tirada do treino
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_treino_total, y_treino,
        test_size=config.PROPORCAO_VALIDACAO,
        stratify=y_treino,
        random_state=config.SEMENTE,
    )

    # 3. Data augmentation (só geometria: não altera as cores)
    augment = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    gerador_treino = augment.flow(
        X_tr, to_categorical(y_tr, num_classes),
        batch_size=BATCH_SIZE, shuffle=True, seed=config.SEMENTE,
    )

    modelo = criar_modelo_cnn(num_classes)
    print("\n--- Estrutura da CNN ---")
    modelo.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    print("\nTreinando a CNN...")
    inicio = time.perf_counter()
    historico = modelo.fit(
        gerador_treino,
        epochs=EPOCHS,
        validation_data=(X_val, to_categorical(y_val, num_classes)),
        callbacks=[early_stopping],
        verbose=2,
    )
    tempo_treino = time.perf_counter() - inicio
    print(f"Tempo de treinamento: {tempo_treino:.1f} s")

    modelo.save(config.CAMINHO_MODELO_CNN)
    print(f"Modelo salvo em: {config.CAMINHO_MODELO_CNN}")

    # 4. Avaliação final no conjunto de TESTE (nunca visto no treino)
    probs = modelo.predict(X_teste, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    avaliar(
        "CNN", y_teste, y_pred, classes, cam_teste,
        extras={
            "tempo_treino_s": tempo_treino,
            "n_parametros": int(modelo.count_params()),
            "tamanho_modelo_mb": os.path.getsize(config.CAMINHO_MODELO_CNN) / 1e6,
            "entrada": "pixels 224x224x3 (150.528 valores)",
            "epocas_treinadas": len(historico.history["loss"]),
            "historico": {k: [float(v) for v in vals] for k, vals in historico.history.items()},
        },
    )
