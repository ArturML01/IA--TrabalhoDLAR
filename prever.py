import os
import numpy as np
import keras

# 1. Parâmetros básicos de configuração
IMAGE_SIZE = (224, 224)
LIMIAR_CONFIANCA = 0.15  # Limiar padrão

def analisar_imagem(imagem_path, modelo, classes):
    img = keras.utils.load_img(imagem_path)
    largura, altura = img.size
    
    print(f"\n🖼️ Analisando imagem: {os.path.basename(imagem_path)} (Dimensões: {largura}x{altura})")
    print("=" * 65)
    
    # --- 1. ANÁLISE GLOBAL ---
    img_global = img.resize(IMAGE_SIZE)
    arr_global = keras.utils.img_to_array(img_global) / 255.0
    arr_global = np.expand_dims(arr_global, axis=0)
    
    preds_global = modelo.predict(arr_global, verbose=0)[0]
    
    print("📊 [CORES DETECTADAS NA IMAGEM INTEIRA]:")
    cores_globais = [(classes[i], preds_global[i]) for i in range(len(classes)) if preds_global[i] >= LIMIAR_CONFIANCA]
    cores_globais = sorted(cores_globais, key=lambda x: x[1], reverse=True)
    
    if cores_globais:
        for cor, prob in cores_globais:
            print(f"   ✔ {cor.upper():<25} ➔ Confiança: {prob * 100:.2f}%")
    else:
        # Fallback global: mostra a mais provável mesmo abaixo do limiar
        idx_max = np.argmax(preds_global)
        print(f"   ⚠️ Nenhuma atingiu {LIMIAR_CONFIANCA*100}%. Mais provável: {classes[idx_max].upper()} ({preds_global[idx_max]*100:.2f}%)")
        
    print("-" * 65)
    
    # --- 2. ANÁLISE POR REGIÕES (GRID 3x3) ---
    print("🔍 [ANÁLISE DETALHADA POR REGIÕES (Grid 3x3)]:")
    
    rows, cols = 3, 3
    step_x = largura / cols
    step_y = altura / rows
    
    todas_cores_encontradas = set()
    
    for r in range(rows):
        for c in range(cols):
            # Garante coordenadas inteiras convertidas para o PIL crop
            left = int(c * step_x)
            upper = int(r * step_y)
            right = int((c + 1) * step_x) if c < cols - 1 else largura
            lower = int((r + 1) * step_y) if r < rows - 1 else altura
            
            box = img.crop((left, upper, right, lower))
            box_resized = box.resize(IMAGE_SIZE)
            
            arr_box = keras.utils.img_to_array(box_resized) / 255.0
            arr_box = np.expand_dims(arr_box, axis=0)
            
            preds_box = modelo.predict(arr_box, verbose=0)[0]
            cores_regiao = [classes[i] for i in range(len(classes)) if preds_box[i] >= LIMIAR_CONFIANCA]
            
            if cores_regiao:
                for cr in cores_regiao:
                    todas_cores_encontradas.add(cr)
                cores_str = ", ".join([c.upper() for c in cores_regiao])
                print(f"   Região [{r+1},{c+1}] ➔ Cores: {cores_str}")
            else:
                # Fallback por região: força mostrar a mais próxima para evitar tela vazia
                idx_max = np.argmax(preds_box)
                cor_provavel = classes[idx_max]
                conf_provavel = preds_box[idx_max]
                todas_cores_encontradas.add(cor_provavel)
                print(f"   Região [{r+1},{c+1}] ➔ [Baixa Confiança] Mais próxima: {cor_provavel.upper()} ({conf_provavel * 100:.2f}%)")
            
    print("-" * 65)
    print(f"🎨 [RESUMO GERAL - Cores únicas mapeadas em toda a imagem]:")
    for cor_unica in sorted(todas_cores_encontradas):
        print(f"   🌟 {cor_unica.upper()}")
    print("=" * 65)

if __name__ == "__main__":
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    
    caminho_modelo = os.path.join(diretorio_script, "imagens_DL", "modelo_treinado.keras")
    dataset_dir = os.path.join(diretorio_script, "imagens_DL", "dataset_geral")
    pasta_teste = os.path.join(diretorio_script, "imagens_teste_cor")
    
    if not os.path.exists(caminho_modelo):
        print(f"[ERRO] O modelo treinado não foi encontrado em: {caminho_modelo}")
        exit(1)
        
    print("Carregando o modelo treinado...")
    modelo = keras.models.load_model(caminho_modelo)
    
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if not os.path.exists(pasta_teste):
        print(f"[ERRO] A pasta de teste '{pasta_teste}' não foi encontrada.")
        exit(1)
        
    extensoes_validas = ('.png', '.jpg', '.jpeg')
    imagens = [f for f in os.listdir(pasta_teste) if f.lower().endswith(extensoes_validas)]
    
    if not imagens:
        print(f"[ERRO] Nenhuma imagem válida encontrada em '{pasta_teste}'.")
        exit(1)
        
    for nome_imagem in imagens:
        imagem_path = os.path.join(pasta_teste, nome_imagem)
        analisar_imagem(imagem_path, modelo, classes)