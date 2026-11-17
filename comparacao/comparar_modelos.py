"""Compara CNN e MLP a partir de resultados/metricas/ e salva os gráficos em resultados/graficos/.

Uso: python comparacao/comparar_modelos.py
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from comum import config
from comum.dados import nome_curto, abrir_imagem

# Estilo dos gráficos
COR = {"CNN": "#2a78d6", "MLP": "#eb6834"}   # par seguro para daltonismo
TEXTO, TEXTO_2, MUDO, GRADE, FUNDO = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
ACERTO, ERRO = "#006300", "#d03b3b"
AZUL_SEQ = LinearSegmentedColormap.from_list(
    "azul", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Amostra de cor ao lado de cada rótulo
AMOSTRA = {
    "branco": "#f7f7f5", "cinza": "#8c8c8c", "preto": "#151515",
    "primaria_amarelo": "#f5d000", "primaria_azul": "#1f5fd6", "primaria_vermelho": "#d62020",
    "secundaria_laranja": "#f28c00", "secundaria_roxo": "#7a2fb5", "secundaria_verde": "#2a9d2a",
    "terciaria_amarelo_laranja": "#f7b500", "terciaria_amarelo_verde": "#9acd32",
    "terciaria_azul_roxo": "#4b3bc4", "terciaria_azul_verde": "#0fa3a3",
    "terciaria_verm_laranja": "#e8501a", "terciaria_verm_roxo": "#b0186b",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "axes.edgecolor": GRADE,
    "axes.labelcolor": TEXTO_2,
    "xtick.color": TEXTO_2,
    "ytick.color": TEXTO_2,
    "text.color": TEXTO,
    "figure.facecolor": FUNDO,
    "axes.facecolor": FUNDO,
    "savefig.facecolor": FUNDO,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def carregar():
    dados = {}
    for nome in ("CNN", "MLP"):
        arq = os.path.join(config.METRICAS_DIR, f"{nome.lower()}.json")
        if not os.path.exists(arq):
            sys.exit(f"[ERRO] {arq} não existe. Rode o treino de {nome} antes.")
        with open(arq, encoding="utf-8") as f:
            dados[nome] = json.load(f)
    return dados


def titulo(fig, principal, subtitulo):
    fig.text(0.04, 0.955, principal, fontsize=26, weight="bold", ha="left", va="top")
    fig.text(0.04, 0.885, subtitulo, fontsize=16, color=TEXTO_2, ha="left", va="top")


def salvar(fig, nome):
    caminho = os.path.join(config.GRAFICOS_DIR, nome)
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"   ✔ {caminho}")


def legenda_modelos(ax, loc="upper left", **kw):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COR["CNN"], label="CNN"), Patch(color=COR["MLP"], label="MLP")],
              loc=loc, frameon=False, fontsize=16, ncol=2, **kw)


def vencedor(valores):
    """Retorna (vencedor, perdedor)."""
    return ("MLP", "CNN") if valores["MLP"] >= valores["CNN"] else ("CNN", "MLP")


# 1. Placar das 4 métricas
def grafico_placar(d):
    nomes = [("acuracia", "Acurácia"), ("precisao", "Precisão"), ("recall", "Recall"), ("f1", "F1-score")]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.76, bottom=0.10)

    x = np.arange(len(nomes))
    larg = 0.36
    for k, modelo in enumerate(("CNN", "MLP")):
        vals = [d[modelo]["metricas"][m] * 100 for m, _ in nomes]
        pos = x + (k - 0.5) * (larg + 0.02)
        ax.bar(pos, vals, width=larg, color=COR[modelo], zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=22, weight="bold")

    ax.set_xticks(x, [n for _, n in nomes], fontsize=20, color=TEXTO)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, color=GRADE, lw=1, zorder=0)
    ax.tick_params(axis="x", length=0)
    ax.spines["left"].set_visible(False)
    legenda_modelos(ax, loc="upper right", bbox_to_anchor=(1, 1.08))

    acc = {m: d[m]["metricas"]["acuracia"] * 100 for m in d}
    v, p = vencedor(acc)
    n = len(d["MLP"]["y_true"])
    acertos = {m: int(np.trace(np.array(d[m]["matriz_confusao"]))) for m in d}
    dif = acertos[v] - acertos[p]
    if acc[v] - acc[p] < 5:
        principal = f"Qualidade quase empatada: {v} {acc[v]:.0f}% × {p} {acc[p]:.0f}% de acurácia"
    else:
        principal = f"{v} acerta {acc[v]:.0f}% das cores, contra {acc[p]:.0f}% da {p}"
    titulo(fig, principal,
           f"Mesmas {n} imagens de teste, nunca vistas no treino (hold-out 70/30) · "
           f"{v} acertou {acertos[v]}, {p} acertou {acertos[p]} ({dif} imagens de diferença)")
    salvar(fig, "01_placar_metricas.png")


# 2. Matrizes de confusão lado a lado
def rotulos_com_amostra(ax, classes, eixo):
    """Rótulo com a amostra da cor ao lado."""
    nomes = [nome_curto(c) for c in classes]
    n = len(classes)
    if eixo == "y":
        ax.set_yticks(range(n), nomes, fontsize=13)
        for i, c in enumerate(classes):
            ax.add_patch(Rectangle((-1.35, i - 0.3), 0.6, 0.6, color=AMOSTRA[c], clip_on=False,
                                   ec=MUDO if c == "branco" else "none", lw=1))
        ax.tick_params(axis="y", pad=40, length=0)
    else:
        ax.set_xticks(range(n), nomes, fontsize=13, rotation=55, ha="right", rotation_mode="anchor")
        for i, c in enumerate(classes):
            ax.add_patch(Rectangle((i - 0.3, n - 0.5 + 0.25), 0.6, 0.6, color=AMOSTRA[c], clip_on=False,
                                   ec=MUDO if c == "branco" else "none", lw=1))
        ax.tick_params(axis="x", pad=40, length=0)


def grafico_matrizes(d):
    classes = d["CNN"]["classes"]
    n = len(classes)
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.81, bottom=0.21, wspace=0.32)

    for ax, modelo in zip(axes, ("CNN", "MLP")):
        m = np.array(d[modelo]["matriz_confusao"])
        pct = m / m.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=AZUL_SEQ, vmin=0, vmax=1)
        for i in range(n):
            for j in range(n):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=12,
                            color="white" if pct[i, j] > 0.5 else TEXTO, weight="bold" if i == j else None)
        # destaca a diagonal (acertos)
        for i in range(n):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=TEXTO, lw=0.6, alpha=0.35))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        for s in ax.spines.values():
            s.set_visible(False)
        rotulos_com_amostra(ax, classes, "y")
        rotulos_com_amostra(ax, classes, "x")
        ax.set_xlabel("Cor prevista pelo modelo", labelpad=8)
        ax.set_ylabel("Cor real", labelpad=8)
        acertos = int(np.trace(m))
        ax.set_title(f"{modelo}  ·  acertou {acertos} de {m.sum()}", fontsize=22, weight="bold",
                     color=COR[modelo], pad=14, loc="left")

    titulo(fig, "Diagonal = acerto. Fora dela = com qual cor o modelo confundiu",
           "Cada linha tem 9 imagens de teste de uma cor; os números contam para onde elas foram classificadas")
    salvar(fig, "02_matrizes_confusao.png")


# 3. F1 por cor (dumbbell)
def grafico_por_classe(d):
    classes = d["CNN"]["classes"]
    f1 = {m: np.array([d[m]["por_classe"][c]["f1-score"] for c in classes]) for m in d}
    ordem = np.argsort(f1["MLP"] - f1["CNN"])          # maior vantagem da MLP no topo
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.subplots_adjust(left=0.20, right=0.95, top=0.80, bottom=0.09)

    for y, i in enumerate(ordem):
        a, b = f1["CNN"][i], f1["MLP"][i]
        empate = np.isclose(a, b)
        ax.plot([a, b], [y, y], color=GRADE, lw=4, zorder=1, solid_capstyle="round")
        # no empate, afasta os pontos para os dois aparecerem
        ax.scatter([a], [y - 0.17 * empate], s=190, color=COR["CNN"], zorder=3, ec=FUNDO, lw=2)
        ax.scatter([b], [y + 0.17 * empate], s=190, color=COR["MLP"], zorder=3, ec=FUNDO, lw=2)
        if empate:
            ax.text(a + 0.025, y, "empate", va="center", fontsize=13, color=TEXTO_2)

    ax.set_yticks(range(len(classes)), [nome_curto(classes[i]) for i in ordem], fontsize=15, color=TEXTO)
    for y, i in enumerate(ordem):
        c = classes[i]
        ax.add_patch(Rectangle((-0.035, y - 0.3), 0.022, 0.6, color=AMOSTRA[c], clip_on=False,
                               transform=ax.get_yaxis_transform(), ec=MUDO if c == "branco" else "none"))
    ax.tick_params(axis="y", pad=34, length=0)
    ax.set_xlim(-0.02, 1.04)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0,25", "0,50", "0,75", "1 (perfeito)"])
    ax.xaxis.grid(True, color=GRADE, lw=1, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("F1-score da cor (equilíbrio entre precisão e recall)")
    legenda_modelos(ax, loc="lower right", bbox_to_anchor=(1, 1.0))

    vantagem = int(np.sum(f1["MLP"] > f1["CNN"]))
    empates = int(np.sum(np.isclose(f1["MLP"], f1["CNN"])))
    cnn_melhor = len(classes) - vantagem - empates
    txt_empate = f"; {empates} empate" + ("s" if empates > 1 else "") if empates else ""
    titulo(fig, f"MLP é melhor em {vantagem} das {len(classes)} cores; CNN em {cnn_melhor}{txt_empate}",
           "Cada linha é uma cor. Quanto mais à direita o ponto, melhor o modelo reconhece aquela cor")
    salvar(fig, "03_f1_por_cor.png")


# 4. Custo computacional (small multiples)
def formatar(v, unidade):
    if unidade == "params":
        txt = f"{v / 1e6:.1f} milhões" if v >= 1e6 else f"{v / 1e3:.0f} mil"
    elif unidade == "s":
        txt = f"{v / 60:.1f} min" if v >= 120 else f"{v:.1f} s"
    elif unidade == "ms":
        txt = f"{v:.1f} ms"
    else:
        txt = f"{v:.1f} MB"
    return txt.replace(".", ",")   # vírgula decimal


def medir_inferencia(d, repeticoes=3):
    """Tempo médio por imagem (ms), com pré-processamento; melhor de `repeticoes` rodadas."""
    sys.path.insert(0, os.path.join(config.RAIZ, "aplicacao"))
    from aplicacao.analise_regioes import carregar_classificador

    imagens = [abrir_imagem(os.path.join(config.DATASET_DIR, c)) for c in d["CNN"]["caminhos_teste"]]
    for nome in ("CNN", "MLP"):
        clf = carregar_classificador(nome)
        clf.prever_proba(imagens[:4])                      # aquecimento
        tempos = []
        for _ in range(repeticoes):
            inicio = time.perf_counter()
            clf.prever_proba(imagens)
            tempos.append((time.perf_counter() - inicio) / len(imagens) * 1000)
        d[nome]["tempo_inferencia_ms"] = min(tempos)


def grafico_custo(d):
    itens = [
        ("tempo_treino_s", "Tempo de treinamento", "s", "mais rápido"),
        ("n_parametros", "Pesos ajustáveis (parâmetros)", "params", "menos pesos"),
        ("tamanho_modelo_mb", "Tamanho do arquivo do modelo", "mb", "menor"),
        ("tempo_inferencia_ms", "Tempo para classificar 1 imagem", "ms", "mais rápido"),
    ]
    fig, axes = plt.subplots(len(itens), 1, figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.72, top=0.78, bottom=0.04, hspace=0.75)

    for ax, (chave, rotulo, un, vantagem) in zip(axes, itens):
        vals = {m: d[m][chave] for m in ("CNN", "MLP")}
        maior = max(vals.values())
        for y, m in enumerate(("CNN", "MLP")):
            ax.barh(y, vals[m], height=0.72, color=COR[m])
            ax.text(vals[m] + maior * 0.01, y, formatar(vals[m], un), va="center", fontsize=16, weight="bold")
        ax.set_yticks([0, 1], ["CNN", "MLP"], fontsize=15)
        ax.invert_yaxis()
        ax.set_xlim(0, maior * 1.2)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(rotulo, loc="left", fontsize=17, weight="bold", color=TEXTO, pad=6)
        lento, rapido = ("CNN", "MLP") if vals["CNN"] >= vals["MLP"] else ("MLP", "CNN")
        razao = vals[lento] / max(vals[rapido], 1e-9)
        txt_razao = f"{razao:,.0f}".replace(",", ".") if razao >= 10 else f"{razao:.1f}".replace(".", ",")
        ax.text(1.03, 0.5, f"{rapido}: {txt_razao}× {vantagem}", transform=ax.transAxes,
                va="center", fontsize=17, color=COR[rapido], weight="bold")

    t = {m: d[m]["tempo_treino_s"] for m in d}
    lento, rapido = ("CNN", "MLP") if t["CNN"] >= t["MLP"] else ("MLP", "CNN")
    titulo(fig, f"{rapido} treina {t[lento] / t[rapido]:,.0f}× mais rápido e é muito mais leve".replace(",", "."),
           f"Mesma máquina (CPU). CNN lê {d['CNN']['entrada']}; MLP lê {d['MLP']['entrada']}")
    fig.text(0.04, 0.012, "Classificar 1 imagem inclui o pré-processamento de cada modelo "
             "(CNN: redimensionar; MLP: extrair os 128 atributos de cor)", fontsize=12, color=MUDO)
    salvar(fig, "04_custo_computacional.png")


# 5. Exemplos reais do teste
def grafico_exemplos(d):
    classes = d["CNN"]["classes"]
    y = np.array(d["CNN"]["y_true"])
    pc, pm = np.array(d["CNN"]["y_pred"]), np.array(d["MLP"]["y_pred"])
    assert d["CNN"]["caminhos_teste"] == d["MLP"]["caminhos_teste"], "Splits diferentes!"
    caminhos = d["CNN"]["caminhos_teste"]

    grupos = [
        np.where((pm == y) & (pc != y))[0],   # só MLP acertou
        np.where((pc == y) & (pm != y))[0],   # só CNN acertou
        np.where((pc == y) & (pm == y))[0],   # ambos acertaram
        np.where((pc != y) & (pm != y))[0],   # ambos erraram
    ]
    cotas = [3, 2, 2, 1]
    rng = np.random.default_rng(config.SEMENTE)
    escolhidos = []
    for g, q in zip(grupos, cotas):
        escolhidos += list(rng.choice(g, size=min(q, len(g)), replace=False)) if len(g) else []
    restantes = [i for i in range(len(y)) if i not in escolhidos]
    while len(escolhidos) < 8 and restantes:
        escolhidos.append(restantes.pop(0))

    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.83, bottom=0.11, wspace=0.10, hspace=0.42)
    for ax, i in zip(axes.ravel(), escolhidos):
        img = abrir_imagem(os.path.join(config.DATASET_DIR, caminhos[i]))
        img.thumbnail((480, 480))
        w, h = img.size
        lado = min(w, h)
        img = img.crop(((w - lado) // 2, (h - lado) // 2, (w + lado) // 2, (h + lado) // 2))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        # cor real + resposta de cada modelo
        ax.text(0.0, -0.05, f"Real: {nome_curto(classes[y[i]])}", transform=ax.transAxes,
                ha="left", va="top", fontsize=16, weight="bold")
        for k, (m, p) in enumerate((("CNN", pc[i]), ("MLP", pm[i]))):
            ok = p == y[i]
            ax.text(0.0, -0.15 - k * 0.09, f"{'✓' if ok else '✗'} {m}: {nome_curto(classes[p])}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=14,
                    color=ACERTO if ok else ERRO, weight="bold")
    for ax in axes.ravel()[len(escolhidos):]:
        ax.axis("off")

    titulo(fig, "Na prática: o que cada modelo respondeu para imagens do teste",
           "✓ verde = acertou   ·   ✗ vermelho = errou (mostrando com qual cor confundiu)")
    salvar(fig, "05_exemplos_teste.png")


# 6. (Apêndice) Curvas de aprendizado
def grafico_curvas(d):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.13, wspace=0.25)

    h = d["CNN"]["historico"]
    ep = np.arange(1, len(h["loss"]) + 1)
    ax = axes[0]
    ax.plot(ep, h["accuracy"], color=COR["CNN"], lw=2.5, label="treino")
    ax.plot(ep, h["val_accuracy"], color=COR["CNN"], lw=2.5, alpha=0.45, label="validação")
    ax.set_title("CNN - acurácia por época", loc="left", weight="bold")
    ax.set_xlabel("Época"); ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, color=GRADE); ax.legend(frameon=False, loc="lower right")
    melhor = int(np.argmin(h["val_loss"])) + 1
    ax.axvline(melhor, color=MUDO, lw=1)
    ax.text(melhor + 0.15, 0.06, f"pesos restaurados\n(época {melhor})", color=TEXTO_2, fontsize=12)

    ax = axes[1]
    perda = d["MLP"]["historico"]["loss"]
    ax.plot(np.arange(1, len(perda) + 1), perda, color=COR["MLP"], lw=2.5)
    ax.set_title("MLP - erro (loss) por iteração", loc="left", weight="bold")
    ax.set_xlabel("Iteração"); ax.set_ylim(0, None)
    ax.yaxis.grid(True, color=GRADE)

    titulo(fig, "Como cada rede aprendeu",
           f"CNN: early stopping parou na época {len(ep)} e restaurou a melhor (época {melhor}). "
           f"MLP: {len(perda)} iterações até o erro estabilizar")
    salvar(fig, "06_curvas_aprendizado.png")


def resumo_terminal(d):
    print("\n" + "=" * 62)
    print(f"{'Métrica':<34}{'CNN':>14}{'MLP':>14}")
    print("-" * 62)
    for chave, nome in (("acuracia", "Acurácia"), ("precisao", "Precisão (macro)"),
                        ("recall", "Recall (macro)"), ("f1", "F1-score (macro)")):
        print(f"{nome:<34}{d['CNN']['metricas'][chave]:>14.4f}{d['MLP']['metricas'][chave]:>14.4f}")
    print(f"{'Tempo de treino (s)':<34}{d['CNN']['tempo_treino_s']:>14.1f}{d['MLP']['tempo_treino_s']:>14.2f}")
    print(f"{'Parâmetros':<34}{d['CNN']['n_parametros']:>14,}{d['MLP']['n_parametros']:>14,}")
    print(f"{'Inferência por imagem (ms)':<34}{d['CNN']['tempo_inferencia_ms']:>14.1f}{d['MLP']['tempo_inferencia_ms']:>14.1f}")
    print("=" * 62)


if __name__ == "__main__":
    dados = carregar()
    print("Medindo tempo de inferência dos dois modelos nas mesmas imagens...")
    medir_inferencia(dados)
    resumo_terminal(dados)
    print("\nGerando gráficos em", config.GRAFICOS_DIR)
    grafico_placar(dados)
    grafico_matrizes(dados)
    grafico_por_classe(dados)
    grafico_custo(dados)
    grafico_exemplos(dados)
    grafico_curvas(dados)
