"""Modelo 2: MLP (scikit-learn). Recebe 128 atributos de cor extraídos da imagem."""
import os

import numpy as np
import joblib
from PIL import Image

from comum import config

TAMANHO_ANALISE = (128, 128)   # reduzida para acelerar os histogramas
BINS_MATIZ = 36                # 1 bin a cada 10 graus


def extrair_atributos(img_pil):
    """Vetor de 128 atributos de cor:
    36 histograma de matiz (pixels coloridos), 8 de saturação, 8 de brilho,
    4 frações preto/branco/cinza/colorido, 6 média e desvio de RGB,
    2 matiz média (cos, sin), 64 histograma RGB 4x4x4.
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
    """Interface usada pela aplicação."""
    nome = "MLP"

    def __init__(self, caminho=config.CAMINHO_MODELO_MLP):
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"Modelo MLP não encontrado em {caminho}. Rode antes: python mlp/treinar_mlp.py")
        self.modelo = joblib.load(caminho)

    def prever_proba(self, imagens_pil):
        X = np.stack([extrair_atributos(img) for img in imagens_pil])
        return self.modelo.predict_proba(X)
