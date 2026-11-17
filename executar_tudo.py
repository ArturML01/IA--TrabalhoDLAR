"""Treina e avalia a CNN e a MLP e gera o comparativo.

Uso: python executar_tudo.py
"""
import os
import sys
import subprocess

RAIZ = os.path.dirname(os.path.abspath(__file__))
ETAPAS = [
    ("CNN", os.path.join("cnn", "treinar_cnn.py")),
    ("MLP", os.path.join("mlp", "treinar_mlp.py")),
    ("COMPARATIVO", os.path.join("comparacao", "comparar_modelos.py")),
]

if __name__ == "__main__":
    for nome, script in ETAPAS:
        print("\n" + "#" * 70 + f"\n#  {nome}\n" + "#" * 70)
        resultado = subprocess.run([sys.executable, script], cwd=RAIZ)
        if resultado.returncode != 0:
            sys.exit(f"[ERRO] A etapa {nome} falhou.")
    print("\nTudo pronto! Gráficos em resultados/graficos/")
