"""
=====================================================================
 MODELO 1 - CNN (Rede Neural Convolucional) - Keras / TensorFlow
=====================================================================
A CNN recebe os PIXELS da imagem (224x224x3) e aprende sozinha, através
dos filtros convolucionais, quais padrões identificam cada cor.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from comum import config

IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)


def criar_modelo_cnn(num_classes):
    model = Sequential([
        Input(shape=INPUT_SHAPE),

        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),

        Dropout(0.5),
        Flatten(),

        Dense(128, activation="relu"),
        # Softmax: cada imagem da base pertence a UMA única cor (pasta),
        # então as probabilidades das 15 classes somam 1.
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def preparar_imagem(img_pil):
    """PIL RGB -> array normalizado (224, 224, 3) em [0, 1]."""
    return keras.utils.img_to_array(img_pil.resize(IMAGE_SIZE)) / 255.0


class ClassificadorCNN:
    """Interface usada pela aplicação (prever / organizar wallpapers)."""
    nome = "CNN"

    def __init__(self, caminho=config.CAMINHO_MODELO_CNN):
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Modelo CNN não encontrado em {caminho}. Rode antes: python cnn/treinar_cnn.py")
        self.modelo = keras.models.load_model(caminho)

    def prever_proba(self, imagens_pil):
        lote = np.stack([preparar_imagem(img) for img in imagens_pil])
        return self.modelo.predict(lote, verbose=0)
