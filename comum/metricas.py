"""Acurácia, precisão, recall, F1 (média macro e por classe) e matriz de confusão."""
import os
import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from comum import config


def avaliar(nome_modelo, y_true, y_pred, classes, caminhos_teste, extras=None):
    """Calcula, imprime e salva as métricas em resultados/metricas/<modelo>.json."""
    rotulos = list(range(len(classes)))

    metricas = {
        "acuracia": accuracy_score(y_true, y_pred),
        "precisao": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    matriz = confusion_matrix(y_true, y_pred, labels=rotulos)
    por_classe = classification_report(
        y_true, y_pred, labels=rotulos, target_names=classes,
        zero_division=0, output_dict=True,
    )

    # Terminal
    print("\n" + "=" * 60)
    print(f"MÉTRICAS NO CONJUNTO DE TESTE (hold-out 30%) - {nome_modelo}")
    print("=" * 60)
    print(f"Acurácia : {metricas['acuracia']:.4f}")
    print(f"Precisão : {metricas['precisao']:.4f}  (macro)")
    print(f"Recall   : {metricas['recall']:.4f}  (macro)")
    print(f"F1-score : {metricas['f1']:.4f}  (macro)")
    print("\nRelatório por classe:")
    print(classification_report(y_true, y_pred, labels=rotulos,
                                target_names=classes, zero_division=0))
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(matriz)

    # Arquivo usado pelo comparativo
    saida = {
        "modelo": nome_modelo,
        "classes": classes,
        "metricas": metricas,
        "por_classe": {c: por_classe[c] for c in classes},
        "matriz_confusao": matriz.tolist(),
        "y_true": np.asarray(y_true).tolist(),
        "y_pred": np.asarray(y_pred).tolist(),
        "caminhos_teste": [os.path.relpath(p, config.DATASET_DIR) for p in caminhos_teste],
    }
    if extras:
        saida.update(extras)

    arquivo = os.path.join(config.METRICAS_DIR, f"{nome_modelo.lower()}.json")
    with open(arquivo, "w", encoding="utf-8") as f:
        json.dump(saida, f, ensure_ascii=False, indent=1, default=float)
    print(f"\nMétricas salvas em: {arquivo}")
    return metricas
