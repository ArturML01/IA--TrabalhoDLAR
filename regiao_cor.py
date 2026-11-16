import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Parâmetros básicos de configuração
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30         
INPUT_SHAPE = (224, 224, 3)
LIMIAR_CONFIANCA = 0.15

def obter_caminho_windows(caminho):
    """Adiciona o prefixo \\?\\ no Windows para contornar o limite de 260 caracteres."""
    abs_path = os.path.abspath(caminho)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        return '\\\\?\\' + abs_path
    return abs_path

def criar_modelo_cnn(num_classes):
    """
    Constrói a arquitetura da CNN para classificação Multi-label usando Keras.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Dropout(0.5),
        Flatten(),
        
        Dense(128, activation='relu'),
        Dense(num_classes, activation='sigmoid') # Sigmoid: Ativa múltiplas cores de forma independente
    ])
    
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[keras.metrics.Precision(name='precision')]
    )
    
    return model

if __name__ == "__main__":
    diretorio_script = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(diretorio_script, "imagens_DL", "dataset_geral")
    
    print(f"Diretório de Dados: {dataset_dir}")
    
    if not os.path.exists(dataset_dir):
        print(f"\n[ERRO] O diretório '{dataset_dir}' não foi encontrado.")
        exit(1)
        
    # --- LIMPEZA AUTOMÁTICA COM SUPORTE A CAMINHOS LONGOS DO WINDOWS ---
    print("Verificando se há arquivos com nomes muito longos...")
    contador_renomeados = 0
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if len(file) > 60: # Nome longo detectado
                caminho_antigo = os.path.join(root, file)
                caminho_antigo_win = obter_caminho_windows(caminho_antigo)
                
                extensao = os.path.splitext(file)[1]
                novo_nome = f"img_cor_{abs(hash(file))}_{contador_renomeados}{extensao}"
                caminho_novo = os.path.join(root, novo_nome)
                caminho_novo_win = obter_caminho_windows(caminho_novo)
                
                try:
                    os.rename(caminho_antigo_win, caminho_novo_win)
                    contador_renomeados += 1
                    print(f"   ✔ Renomeado com sucesso: {file[:30]}... ➔ {novo_nome}")
                except Exception as e:
                    try:
                        os.remove(caminho_antigo_win)
                        contador_renomeados += 1
                        print(f"   🗑️ Removido arquivo inacessível: {file[:30]}...")
                    except Exception as e2:
                        print(f"   [AVISO] Não foi possível tratar {file[:20]}: {e2}")
                        
    if contador_renomeados > 0:
        print(f"✔ Total de arquivos ajustados/removidos: {contador_renomeados}")
    else:
        print("✔ Nenhum arquivo com nome excessivamente longo encontrado.")

    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    num_classes = len(classes)
    print(f"Classes detectadas ({num_classes}): {classes}")
    
    if num_classes == 0:
        print("\n[ERRO] Nenhuma subpasta de classe encontrada em 'dataset_geral'.")
        exit(1)
    
    # 2. Configurando o Data Augmentation separado para Treino e Validação
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.3,
        rotation_range=20,         
        width_shift_range=0.2,     
        height_shift_range=0.2,    
        shear_range=0.2,           
        zoom_range=0.2,            
        horizontal_flip=True,      
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.3
    )
    
    train_generator = train_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        dataset_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"\n-> Treinamento: {train_generator.samples} | Validação: {validation_generator.samples}")
    
    if train_generator.samples == 0 or validation_generator.samples == 0:
        print("\n[ERRO] Adicione imagens dentro das subpastas em 'dataset_geral'.")
        exit(1)
    
    # Instancia o modelo
    modelo = criar_modelo_cnn(num_classes)
    print("\n--- Estrutura da Rede Neural Convolucional (Multi-label) ---")
    modelo.summary()
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    print("\nIniciando o treinamento com Data Augmentation e Early Stopping...")
    history = modelo.fit(
        train_generator,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )
    
    print("\nTreinamento concluído com sucesso!")
    
    # Salvando o modelo
    caminho_modelo = os.path.join(diretorio_script, "imagens_DL", "modelo_treinado.keras")
    modelo.save(caminho_modelo)
    print(f"Modelo salvo em: {caminho_modelo}")

    # --- AVALIAÇÃO FINAL COM SCIKIT-LEARN ---
    print("\n" + "="*50)
    print("GERANDO MÉTRICAS DE AVALIAÇÃO (Validação)")
    print("="*50)
    
    validation_generator.reset()
    
    # 1. Fazendo predições de probabilidade no conjunto de validação
    y_pred_probs = modelo.predict(validation_generator, verbose=1)
    
    # 2. Aplicando o limiar de confiança (Threshold) para binarizar as predições multi-label
    y_pred_binario = (y_pred_probs >= LIMIAR_CONFIANCA).astype(int)
    
    # 3. Obtendo os rótulos reais (ground truth) em formato one-hot
    y_true = to_categorical(validation_generator.classes, num_classes=num_classes)
    
    print("\nRelatório de Classificação (Precision, Recall, F1-Score por classe):")
    print(classification_report(y_true, y_pred_binario, target_names=classes, zero_division=0))
    