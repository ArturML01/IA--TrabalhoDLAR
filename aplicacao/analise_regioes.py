"""
Lógica da aplicação, comum aos dois modelos: analisa a imagem inteira e uma
grade 3x3 de regiões para descobrir as cores presentes num wallpaper.

O modelo (CNN ou MLP) só precisa oferecer o método prever_proba(lista_de_imagens_PIL).
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import numpy as np


def carregar_classificador(nome):
    nome = nome.lower()
    if nome == "cnn":
        from cnn.modelo_cnn import ClassificadorCNN
        return ClassificadorCNN()
    if nome == "mlp":
        from mlp.modelo_mlp import ClassificadorMLP
        return ClassificadorMLP()
    raise ValueError("Modelo deve ser 'cnn' ou 'mlp'")


def recortar_grade(img, linhas=3, colunas=3):
    """Divide a imagem em uma grade linhas x colunas e devolve [(linha, coluna, recorte)]."""
    largura, altura = img.size
    passo_x, passo_y = largura / colunas, altura / linhas
    recortes = []
    for r in range(linhas):
        for c in range(colunas):
            left, upper = int(c * passo_x), int(r * passo_y)
            right = int((c + 1) * passo_x) if c < colunas - 1 else largura
            lower = int((r + 1) * passo_y) if r < linhas - 1 else altura
            recortes.append((r, c, img.crop((left, upper, right, lower))))
    return recortes


def analisar(img, classificador):
    """Retorna (probabilidades_imagem_inteira, [(linha, coluna, probabilidades_regiao)])."""
    recortes = recortar_grade(img)
    probs = classificador.prever_proba([img] + [rec for _, _, rec in recortes])
    regioes = [(r, c, p) for (r, c, _), p in zip(recortes, probs[1:])]
    return probs[0], regioes


def cor_principal(img, classificador, classes, limiar):
    """Cor com maior confiança entre as regiões; se nenhuma passa o limiar,
    usa a mais provável da imagem inteira."""
    probs_global, regioes = analisar(img, classificador)
    melhores = {}
    for _, _, p in regioes:
        for i, prob in enumerate(p):
            if prob >= limiar and prob > melhores.get(classes[i], 0):
                melhores[classes[i]] = prob
    if melhores:
        return max(melhores, key=melhores.get)
    return classes[int(np.argmax(probs_global))]
