"""Leitura da base (imagens_DL/dataset_geral/<cor>/) e divisão hold-out."""
import os
import json

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from comum import config


def caminho_longo(caminho):
    """Prefixo \\\\?\\ para caminhos com mais de 260 caracteres no Windows."""
    abs_path = os.path.abspath(caminho)
    if os.name == "nt" and not abs_path.startswith("\\\\?\\"):
        return "\\\\?\\" + abs_path
    return abs_path


def abrir_imagem(caminho):
    """Abre qualquer imagem como RGB."""
    with Image.open(caminho_longo(caminho)) as img:
        return img.convert("RGB")


def listar_classes():
    return sorted(
        d for d in os.listdir(config.DATASET_DIR)
        if os.path.isdir(os.path.join(config.DATASET_DIR, d))
    )


def listar_base():
    """Retorna (caminhos, rotulos_inteiros, nomes_das_classes)."""
    classes = listar_classes()
    caminhos, rotulos = [], []
    for idx, classe in enumerate(classes):
        pasta = os.path.join(config.DATASET_DIR, classe)
        for arquivo in sorted(os.listdir(pasta)):
            if arquivo.lower().endswith(config.EXTENSOES_VALIDAS):
                caminhos.append(os.path.join(pasta, arquivo))
                rotulos.append(idx)
    return np.array(caminhos), np.array(rotulos), classes


def dividir_holdout():
    """Hold-out estratificado 70/30, salvo em disco para CNN e MLP usarem as mesmas imagens."""
    caminhos, rotulos, classes = listar_base()
    arquivo_split = os.path.join(config.CACHE_DIR, "divisao_holdout.json")

    idx = np.arange(len(caminhos))
    idx_treino, idx_teste = train_test_split(
        idx,
        test_size=config.PROPORCAO_TESTE,
        stratify=rotulos,
        random_state=config.SEMENTE,
    )

    with open(arquivo_split, "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "treino": [os.path.relpath(p, config.DATASET_DIR) for p in caminhos[idx_treino]],
            "teste": [os.path.relpath(p, config.DATASET_DIR) for p in caminhos[idx_teste]],
        }, f, ensure_ascii=False, indent=1)

    return (caminhos[idx_treino], rotulos[idx_treino],
            caminhos[idx_teste], rotulos[idx_teste], classes)


def nome_curto(classe):
    """'terciaria_azul_verde' -> 'azul-verde'."""
    partes = classe.split("_")
    if partes[0] in ("primaria", "secundaria", "terciaria"):
        partes = partes[1:]
    return "-".join("vermelho" if p == "verm" else p for p in partes)
