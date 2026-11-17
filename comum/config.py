"""
Configurações compartilhadas entre a CNN e a MLP.

Os dois modelos usam EXATAMENTE a mesma divisão hold-out (mesma semente e as
mesmas imagens de teste) para que a comparação seja justa.
"""
import os

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dados
DATASET_DIR = os.path.join(RAIZ, "imagens_DL", "dataset_geral")
PASTA_WALLPAPERS = os.path.join(RAIZ, "imagens_teste_cor")
EXTENSOES_VALIDAS = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp")

# Hold-out
PROPORCAO_TESTE = 0.30       # 70% treino / 30% teste
PROPORCAO_VALIDACAO = 0.15   # parte do TREINO usada só para o early stopping da CNN
SEMENTE = 42

# Saídas
RESULTADOS_DIR = os.path.join(RAIZ, "resultados")
MODELOS_DIR = os.path.join(RESULTADOS_DIR, "modelos")
METRICAS_DIR = os.path.join(RESULTADOS_DIR, "metricas")
GRAFICOS_DIR = os.path.join(RESULTADOS_DIR, "graficos")
CACHE_DIR = os.path.join(RESULTADOS_DIR, "cache")
WALLPAPERS_ORGANIZADOS_DIR = os.path.join(RESULTADOS_DIR, "wallpapers_organizados")

CAMINHO_MODELO_CNN = os.path.join(MODELOS_DIR, "cnn.keras")
CAMINHO_MODELO_MLP = os.path.join(MODELOS_DIR, "mlp.joblib")

for _d in (MODELOS_DIR, METRICAS_DIR, GRAFICOS_DIR, CACHE_DIR, WALLPAPERS_ORGANIZADOS_DIR):
    os.makedirs(_d, exist_ok=True)
