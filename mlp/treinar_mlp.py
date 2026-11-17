"""
=====================================================================
 TREINAMENTO E AVALIAÇÃO DA MLP (sklearn.neural_network.MLPClassifier)
=====================================================================
Uso (a partir da pasta do projeto):
    python mlp/treinar_mlp.py

Validação: Hold-out estratificado 70% treino / 30% teste (mesmo split da CNN).
Os hiperparâmetros (neurônios por camada e regularização) são escolhidos
com validação cruzada de 5 folds usando SOMENTE os 70% de treino.
O conjunto de TESTE só é usado no final, para as métricas.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from comum import config
from comum.dados import dividir_holdout, abrir_imagem
from comum.metricas import avaliar
from mlp.modelo_mlp import extrair_atributos


def carregar_atributos(caminhos):
    return np.stack([extrair_atributos(abrir_imagem(p)) for p in caminhos])


def criar_pipeline():
    return Pipeline([
        ("padronizacao", StandardScaler()),   # média 0 / desvio 1 em cada atributo
        ("mlp", MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=config.SEMENTE,
        )),
    ])


GRADE_HIPERPARAMETROS = {
    "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
    "mlp__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
}


if __name__ == "__main__":
    # 1. Mesma divisão hold-out usada pela CNN
    cam_treino, y_treino, cam_teste, y_teste, classes = dividir_holdout()
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Treino: {len(cam_treino)} imagens | Teste: {len(cam_teste)} imagens")

    print("Extraindo atributos de cor das imagens...")
    inicio = time.perf_counter()
    X_treino = carregar_atributos(cam_treino)
    X_teste = carregar_atributos(cam_teste)
    print(f"Vetor de atributos: {X_treino.shape[1]} valores por imagem "
          f"({time.perf_counter() - inicio:.1f} s)")

    # 2. Escolha de hiperparâmetros com validação cruzada (só no treino)
    print("\nBuscando a melhor arquitetura da MLP (5-fold CV no treino)...")
    busca = GridSearchCV(
        criar_pipeline(),
        GRADE_HIPERPARAMETROS,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEMENTE),
        scoring="f1_macro",
        n_jobs=-1,
    )
    inicio = time.perf_counter()
    busca.fit(X_treino, y_treino)
    tempo_busca = time.perf_counter() - inicio
    print(f"Melhores hiperparâmetros: {busca.best_params_} "
          f"(F1 médio na CV: {busca.best_score_:.4f}) - busca levou {tempo_busca:.1f} s")

    # 3. Treino final com a melhor configuração (cronometrado)
    modelo = criar_pipeline().set_params(**busca.best_params_)
    inicio = time.perf_counter()
    modelo.fit(X_treino, y_treino)
    tempo_treino = time.perf_counter() - inicio
    print(f"Tempo de treinamento: {tempo_treino:.2f} s")

    mlp = modelo.named_steps["mlp"]
    print("\n--- Estrutura da MLP ---")
    camadas = [X_treino.shape[1], *mlp.hidden_layer_sizes, len(classes)]
    print("Camadas: " + " -> ".join(str(c) for c in camadas))
    n_parametros = int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
    print(f"Total de parâmetros: {n_parametros:,}")

    joblib.dump(modelo, config.CAMINHO_MODELO_MLP)
    print(f"Modelo salvo em: {config.CAMINHO_MODELO_MLP}")

    # 4. Avaliação final no conjunto de TESTE
    y_pred = modelo.predict(X_teste)

    avaliar(
        "MLP", y_teste, y_pred, classes, cam_teste,
        extras={
            "tempo_treino_s": tempo_treino,
            "tempo_busca_hiperparametros_s": tempo_busca,
            "n_parametros": n_parametros,
            "tamanho_modelo_mb": os.path.getsize(config.CAMINHO_MODELO_MLP) / 1e6,
            "entrada": f"{X_treino.shape[1]} atributos de cor",
            "camadas": camadas,
            "melhores_hiperparametros": {k: str(v) for k, v in busca.best_params_.items()},
            "epocas_treinadas": int(mlp.n_iter_),
            "historico": {"loss": [float(v) for v in mlp.loss_curve_]},
        },
    )
