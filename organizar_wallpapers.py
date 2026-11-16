import os
import shutil
import numpy as np
import keras

IMAGE_SIZE = (224, 224)
LIMIAR_CONFIANCA = 0.25

def organizar_pasta_wallpapers():
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    caminho_modelo = os.path.join(diretorio_script, "imagens_DL", "modelo_treinado.keras")
    dataset_dir = os.path.join(diretorio_script, "imagens_DL", "dataset_geral")
    
    pasta_origem = os.path.join(diretorio_script, "imagens_teste_cor") 
    pasta_destino = os.path.join(diretorio_script, "wallpapers_organizados")
    
    if not os.path.exists(caminho_modelo):
        print(f"[ERRO] Modelo não encontrado em: {caminho_modelo}")
        return
        
    print("Carregando modelo treinado...")
    modelo = keras.models.load_model(caminho_modelo)
    
    # Pega a lista exata das 15 classes treinadas
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    extensoes_validas = ('.png', '.jpg', '.jpeg', '.webp')
    imagens = [f for f in os.listdir(pasta_origem) if f.lower().endswith(extensoes_validas)]
    
    if not imagens:
        print(f"[AVISO] Nenhuma imagem encontrada em '{pasta_origem}'.")
        return
        
    print(f"Iniciando organização de {len(imagens)} wallpapers...")
    
    for nome_imagem in imagens:
        caminho_img = os.path.join(pasta_origem, nome_imagem)
        
        try:
            img = keras.utils.load_img(caminho_img)
            largura, altura = img.size
            
            # --- Análise simplificada por Grid 3x3 para pegar as cores ---
            rows, cols = 3, 3
            step_x = largura / cols
            step_y = altura / rows
            
            cores_detectadas = {} 
            
            for r in range(rows):
                for c in range(cols):
                    left = int(c * step_x)
                    upper = int(r * step_y)
                    right = int((c + 1) * step_x) if c < cols - 1 else largura
                    lower = int((r + 1) * step_y) if r < rows - 1 else altura
                    
                    box = img.crop((left, upper, right, lower)).resize(IMAGE_SIZE)
                    arr_box = keras.utils.img_to_array(box) / 255.0
                    arr_box = np.expand_dims(arr_box, axis=0)
                    
                    preds = modelo.predict(arr_box, verbose=0)[0]
                    
                    for i, prob in enumerate(preds):
                        if prob >= LIMIAR_CONFIANCA:
                            nome_classe = classes[i]
                            if nome_classe not in cores_detectadas or prob > cores_detectadas[nome_classe]:
                                cores_detectadas[nome_classe] = prob
                                
            if cores_detectadas:
                # Escolhe a cor com maior probabilidade média/máxima detectada nas regiões
                cor_principal = max(cores_detectadas, key=cores_detectadas.get)
            else:
                # Fallback: se nenhuma bateu o limiar, pega a absoluta mais provável da imagem inteira
                img_global = img.resize(IMAGE_SIZE)
                arr_global = keras.utils.img_to_array(img_global) / 255.0
                preds_global = modelo.predict(np.expand_dims(arr_global, axis=0), verbose=0)[0]
                cor_principal = classes[np.argmax(preds_global)]
                
            # Cria a pasta de destino para essa cor específica
            pasta_cor = os.path.join(pasta_destino, cor_principal)
            os.makedirs(pasta_cor, exist_ok=True)
            
            # Copia a imagem para a pasta organizada
            destino_final = os.path.join(pasta_cor, nome_imagem)
            shutil.copy(caminho_img, destino_final)
            print(f"   ✔ {nome_imagem} ➔ {cor_principal}")
            
        except Exception as e:
            print(f"   [ERRO] Falha ao processar {nome_imagem}: {e}")
            
    print("\n🎉 Organização de wallpapers concluída com sucesso!")

if __name__ == "__main__":
    organizar_pasta_wallpapers()