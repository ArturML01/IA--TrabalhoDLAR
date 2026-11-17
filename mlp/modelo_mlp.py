"""
=====================================================================
 MODELO 2 - MLP (Perceptron de Múltiplas Camadas) - scikit-learn
=====================================================================
A MLP não enxerga a imagem diretamente. Primeiro cada imagem é resumida
em um vetor de 128 ATRIBUTOS DE COR (histogramas de matiz, saturação,
brilho, médias RGB...). A MLP (sklearn.neural_network.MLPClassifier)
aprende a classificar a cor a partir desse vetor.
"""
import os

import numpy as np
import joblib
from PIL import Image

from comum import config

TAMANHO_ANALISE = (128, 128)   # reduz a imagem só para acelerar os histogramas
BINS_MATIZ = 36                # 1 bin a cada 10 graus do círculo cromático


def extrair_atributos(img_pil):
    """
    Converte uma imagem em um vetor de 128 atributos de cor:
      - 36  histograma de MATIZ (só pixels coloridos)     -> "qual cor"
      -  8  histograma de SATURAÇÃO                       -> "quão viva"
      -  8  histograma de BRILHO (V)                      -> "quão clara/escura"
      -  4  fração de pixels preto / branco / cinza / colorido
      -  6  média e desvio padrão de R, G, B
      -  2  matiz média (cos, sin) dos pixels coloridos
      - 64  histograma RGB conjunto 4x4x4
    """
    img = img_pil.convert("RGB").resize(TAMANHO_ANALISE, Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    h, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    n = h.size

    colorido = (s > 0.15) & (v > 0.15)
    escuro = v <= 0.15
    claro_neutro = (~colorido) & (~escuro) & (v > 0.80)
    cinza = (~colorido) & (~escuro) & (v <= 0.80)

    hist_h = np.histogram(h[colorido], bins=BINS_MATIZ, range=(0, 1))[0] / n
    hist_s = np.histogram(s, bins=8, range=(0, 1))[0] / n
    hist_v = np.histogram(v, bins=8, range=(0, 1))[0] / n
    fracoes = np.array([escuro.mean(), claro_neutro.mean(), cinza.mean(), colorido.mean()])

    pix = rgb.reshape(-1, 3)
    estat_rgb = np.concatenate([pix.mean(axis=0), pix.std(axis=0)])

    if colorido.any():
        ang = h[colorido] * 2 * np.pi
        matiz_media = np.array([np.cos(ang).mean(), np.sin(ang).mean()])
    else:
        matiz_media = np.zeros(2)

    q = np.minimum((pix * 4).astype(int), 3)
    hist_rgb = np.bincount(q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2], minlength=64) / n

    return np.concatenate([hist_h, hist_s, hist_v, fracoes, estat_rgb, matiz_media, hist_rgb]).astype(np.float32)


class ClassificadorMLP:
    """Interface usada pela aplicação (prever / organizar wallpapers)."""
    nome = "MLP"

    def __init__(self, caminho=config.CAMINHO_MODELO_MLP):
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Modelo MLP não encontrado em {caminho}. Rode antes: python mlp/treinar_mlp.py")
        self.modelo = joblib.load(caminho)

    def prever_proba(self, imagens_pil):
        X = np.stack([extrair_atributos(img) for img in imagens_pil])
        return self.modelo.predict_proba(X)
