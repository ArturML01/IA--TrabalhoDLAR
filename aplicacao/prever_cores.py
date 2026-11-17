"""Mostra as cores de cada wallpaper de imagens_teste_cor/ (imagem inteira + grade 3x3).

Uso: python aplicacao/prever_cores.py --modelo {cnn,mlp}
"""
import os
import argparse

from analise_regioes import carregar_classificador, analisar
from comum import config
from comum.dados import listar_classes, abrir_imagem

LIMIAR_CONFIANCA = 0.15


def mostrar(nome_arquivo, img, classificador, classes):
    probs_global, regioes = analisar(img, classificador)
    print(f"\n🖼️ {nome_arquivo} ({img.size[0]}x{img.size[1]}) - modelo {classificador.nome}")
    print("=" * 65)

    print("📊 [IMAGEM INTEIRA]:")
    detectadas = sorted(((classes[i], p) for i, p in enumerate(probs_global) if p >= LIMIAR_CONFIANCA),
                        key=lambda x: x[1], reverse=True)
    if not detectadas:
        i = int(probs_global.argmax())
        detectadas = [(classes[i], probs_global[i])]
    for cor, p in detectadas:
        print(f"   ✔ {cor.upper():<27} ➔ {p * 100:.1f}%")

    print("🔍 [REGIÕES 3x3]:")
    todas = {c for c, _ in detectadas}
    for r, c, p in regioes:
        cores = [classes[i] for i, prob in enumerate(p) if prob >= LIMIAR_CONFIANCA]
        if not cores:
            cores = [classes[int(p.argmax())]]
        todas.update(cores)
        print(f"   Região [{r + 1},{c + 1}] ➔ {', '.join(x.upper() for x in cores)}")

    print("🎨 [CORES MAPEADAS]: " + ", ".join(sorted(x.upper() for x in todas)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=["cnn", "mlp"], default="mlp")
    args = parser.parse_args()

    classificador = carregar_classificador(args.modelo)
    classes = listar_classes()
    for arquivo in sorted(os.listdir(config.PASTA_WALLPAPERS)):
        if arquivo.lower().endswith(config.EXTENSOES_VALIDAS):
            img = abrir_imagem(os.path.join(config.PASTA_WALLPAPERS, arquivo))
            mostrar(arquivo, img, classificador, classes)
