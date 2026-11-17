"""Copia cada wallpaper de imagens_teste_cor/ para resultados/wallpapers_organizados/<modelo>/<cor>/.

Uso: python aplicacao/organizar_wallpapers.py --modelo {cnn,mlp}
"""
import os
import shutil
import argparse

from analise_regioes import carregar_classificador, cor_principal
from comum import config
from comum.dados import listar_classes, abrir_imagem

LIMIAR_CONFIANCA = 0.25


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", choices=["cnn", "mlp"], default="mlp")
    args = parser.parse_args()

    classificador = carregar_classificador(args.modelo)
    classes = listar_classes()
    destino = os.path.join(config.WALLPAPERS_ORGANIZADOS_DIR, args.modelo)

    imagens = sorted(f for f in os.listdir(config.PASTA_WALLPAPERS)
                     if f.lower().endswith(config.EXTENSOES_VALIDAS))
    print(f"Organizando {len(imagens)} wallpapers com a {classificador.nome}...")

    for arquivo in imagens:
        origem = os.path.join(config.PASTA_WALLPAPERS, arquivo)
        try:
            cor = cor_principal(abrir_imagem(origem), classificador, classes, LIMIAR_CONFIANCA)
            pasta_cor = os.path.join(destino, cor)
            os.makedirs(pasta_cor, exist_ok=True)
            shutil.copy(origem, os.path.join(pasta_cor, arquivo))
            print(f"   ✔ {arquivo} ➔ {cor}")
        except Exception as e:
            print(f"   [ERRO] {arquivo}: {e}")

    print(f"\n🎉 Concluído! Veja: {destino}")
