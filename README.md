# Classificação de cores predominantes em wallpapers — CNN × MLP

Duas redes neurais treinadas na **mesma base** e avaliadas nas **mesmas imagens de teste**,
para comparar uma **CNN** (Keras, lê os pixels) com uma **MLP** (`sklearn.neural_network.MLPClassifier`,
lê atributos de cor).

- **Problema:** classificação multiclasse, 15 cores (branco, cinza, preto, primárias, secundárias e terciárias).
- **Base:** 450 imagens próprias (30 por cor) em `imagens_DL/dataset_geral/<cor>/`.
- **Validação:** Hold-out estratificado **70% treino / 30% teste** (`random_state=42`), idêntico para os dois modelos.
- **Métricas:** acurácia, precisão, recall, F1-score (média macro e por cor) e matriz de confusão.

## Estrutura do código

```
cnn/                       ← MODELO 1: CNN (Keras/TensorFlow)
  modelo_cnn.py              arquitetura + carregamento do modelo
  treinar_cnn.py             treino + avaliação no teste
mlp/                       ← MODELO 2: MLP (scikit-learn)
  modelo_mlp.py              extração dos 128 atributos de cor + carregamento
  treinar_mlp.py             busca de hiperparâmetros (5-fold CV no treino) + treino + avaliação
comum/                     ← compartilhado pelos dois (garante comparação justa)
  config.py                  caminhos, semente, proporções do hold-out
  dados.py                   leitura da base e divisão hold-out
  metricas.py                as 5 métricas exigidas
comparacao/
  comparar_modelos.py        tabela comparativa + gráficos em resultados/graficos/
aplicacao/                 ← uso prático, funciona com qualquer um dos modelos
  prever_cores.py            cores de cada wallpaper (imagem inteira + grade 3x3)
  organizar_wallpapers.py    copia cada wallpaper para a pasta da sua cor
executar_tudo.py           roda CNN → MLP → comparativo
```

## Como executar

```bash
pip install -r requirements.txt        # Python 3.10 (TensorFlow ainda não suporta 3.14)
python executar_tudo.py                # treina os dois, avalia e gera os gráficos

# Aplicação (escolha o modelo)
python aplicacao/prever_cores.py --modelo mlp
python aplicacao/organizar_wallpapers.py --modelo cnn
```

## Resultados (135 imagens de teste)

| Métrica                     | CNN          | MLP         |
|-----------------------------|--------------|-------------|
| Acurácia                    | 74,8%        | **77,8%**   |
| Precisão (macro)            | **80,6%**    | 78,3%       |
| Recall (macro)              | 74,8%        | **77,8%**   |
| F1-score (macro)            | 74,3%        | **77,5%**   |
| Acertos                     | 101          | **105**     |
| Tempo de treino (CPU)       | ~60–75 s     | **~1 s**    |
| Parâmetros                  | 23,9 milhões | **25,7 mil**|
| Tamanho do modelo           | 287 MB       | **0,4 MB**  |
| Tempo p/ classificar 1 imagem | ~6–7 ms    | **~4 ms**   |

Arquiteturas:
- **CNN:** Conv(32) → Pool → Conv(64) → Pool → Dropout → Dense(128) → Softmax(15); data augmentation geométrico; early stopping com validação separada do treino.
- **MLP:** 128 atributos → 128 → 64 → 15 neurônios (ReLU, Adam, `alpha=0.01`), escolhida por GridSearch com 5-fold CV no treino.

## Roteiro sugerido (5 min)

| Tempo | Slide / gráfico | Mensagem |
|---|---|---|
| 0:00–0:45 | Problema | 450 wallpapers, 15 cores, hold-out 70/30 igual para os dois modelos |
| 0:45–1:30 | Arquiteturas | CNN olha os pixels (150 mil valores); MLP olha 128 atributos de cor |
| 1:30–2:15 | `01_placar_metricas.png` | Qualidade quase empatada (78% × 75%); CNN tem a maior precisão |
| 2:15–3:00 | `02_matrizes_confusao.png` | Os dois erram as **cores vizinhas** (cinza↔branco/preto, laranja↔vermelho-laranja) |
| 3:00–3:30 | `03_f1_por_cor.png` | MLP vence em 11 cores, CNN em 3, 1 empate |
| 3:30–4:15 | `04_custo_computacional.png` | MLP treina ~60× mais rápido com ~900× menos parâmetros |
| 4:15–5:00 | `05_exemplos_teste.png` + conclusão | Para cor, atributos bem escolhidos + MLP bastam; a CNN gasta capacidade aprendendo formas que não importam aqui |

`06_curvas_aprendizado.png` é apêndice, para o caso de perguntarem sobre overfitting ou convergência.

**Limitação:** o teste tem só 9 imagens por cor, então 4 acertos de diferença (3 p.p.) não é uma diferença estatisticamente forte. A vantagem clara da MLP está no custo.
