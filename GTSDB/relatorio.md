# Relatório Técnico — Traffic Sign Recognition (GTSDB)

**Grupo:** A52323 | **Split:** 70/15/15 estratificado, seed=42 → 849 / 182 / 182

---

## Resultados e Log de Execução

### T1 — CNN from Scratch (4 super-classes, patches 96×96)

| Métrica | Valor |
|---------|-------|
| Accuracy (test) | **98.9%** (180/182 corretos) |
| Macro F1 | **0.9871** |
| AUC (todas as classes) | **1.000** |
| Parâmetros | 4,814,020 |
| Épocas | 33 (early stop patience=10) |

**Erros (2 casos):**
- Mandatory → Other: sinal de rotunda (setas circulares azuis) confundido com Other
- Other → Prohibitory: sinal "fim de todas as restrições" (classe 32, bordas circulares vermelhas) confundido com Prohibitory

---

### T2 — MobileNetV2 (43 classes, patches 96×96)

#### Run 1 — BROKEN ❌

| Métrica | Valor |
|---------|-------|
| Accuracy (test) | **1.65%** (pior que random 1/43 ≈ 2.3%) |
| Macro F1 | **0.0032** |
| Classes aprendidas | 2 de 43 (classe 1: F1≈0.09, classe 20: F1≈0.02) |

**Causa raiz:** `class_weight_dict` com pesos extremos (~19.7×) para classes raras (19 e 37).
Com 2 amostras a pesar 19.7× cada, essas amostras dominam os gradientes de toda a época.
Os gradientes das outras 847 amostras ficam suprimidos. Loss estagna em ~3.76 (entropy máxima para log(43)).

**Fix:**
```python
max_weight = 5.0
class_weight_capped = {k: min(v, max_weight) for k, v in class_weight_dict.items()}
```
Reconstruir o modelo + re-executar Phase 1 + Phase 2 com `class_weight_capped`.

#### Run 2 — class_weight capped ❌ (Accuracy: 1.65% | Macro F1: 0.003)

Resultados idênticos à run 1. O cap não foi suficiente — o problema não era só o class_weight.

#### Runs 3-4 ⚠️ (Accuracy: 6.59% | Macro F1: 0.003)

Class collapse — modelo prevê sempre a classe mais frequente. Causa ainda não identificada nestas runs.

#### Run 5 — preprocessing corrigido ✅ (Accuracy: 36.26% | Macro F1: 0.196)

**Bug identificado:** pipeline divide imagens por 255 → [0,1]. Modelo aplicava `mobilenet_v2.preprocess_input` que espera [0,255] → todos os inputs comprimidos em [-1.0, -0.992]. Backbone recebia inputs virtualmente idênticos.

**Fix:** `x = inputs * 2.0 - 1.0` — mapeia [0,1] → [-1,1] corretamente.

- Phase 1 (frozen): val_accuracy 2.2% → **36.8%**
- Phase 2 (20 camadas, lr=1e-5): test accuracy **36.26%**, macro F1 **0.196**
- Classes bem aprendidas (F1>0.5): 6, 12, 13, 14
- Classes F1=0: raras com 1-4 amostras de treino

---

### Tabela Comparativa

| Task | Accuracy | Macro F1 | Parâmetros |
|------|----------|----------|------------|
| T1: CNN scratch (4 classes) | **0.9890** | **0.9871** | 4,814,020 |
| T2 run1 (43 classes) | 0.0165 ❌ | 0.0032 ❌ | 2,596,971 |
| T2 run2 (43 classes) | 0.0165 ❌ | 0.0032 ❌ | 2,596,971 |
| T2 run3 (43 classes) | 0.0659 ⚠️ | 0.0034 ❌ | 2,596,971 |
| T2 run4 — head simples + aug++ | 0.0659 ⚠️ | 0.0034 ❌ | 2,313,067 |
| T2 run5 — MobileNetV2 preprocessing fix ✅ | **0.3626** | **0.1963** | 2,313,067 |
| T2 run6 — EfficientNetB0 | ~0.17 ❌ | — | ~4,000,000 |
| T3: Multi-label | — | — | — |
| T4: YOLO detection | — | — | — |

---

### Problemas e Soluções

| Problema | Causa | Fix |
|----------|-------|-----|
| `NotFoundError: dlopen libmetal_plugin.dylib` | tensorflow-metal 1.2.0 incompatível com TF 2.20.0 | `pip uninstall tensorflow-metal -y` + eliminar célula de instalação |
| T2 run1: accuracy < random | class_weight extremo (19.7×) suprime gradientes | Limitar pesos a `min(w, 5.0)` |
| T2 runs 1-4: class collapse (6.59%) | **Bug de preprocessing**: `preprocess_input` esperava [0,255] mas recebia [0,1] → todos os inputs em [-1,-0.99] | `x = inputs * 2.0 - 1.0` — fix na run 5 → 36.26% |

---

## Índice

1. [Visão geral do projeto](#1-visão-geral-do-projeto)
2. [Pré-processamento](#2-pré-processamento)
3. [Task 1 — CNN from Scratch](#3-task-1--cnn-from-scratch)
4. [Task 2 — Transfer Learning (43 classes)](#4-task-2--transfer-learning-43-classes)
5. [Task 3 — Multi-label Classification](#5-task-3--multi-label-classification)
6. [Task 4 — Object Detection com YOLOv8](#6-task-4--object-detection-com-yolov8)
7. [Decisões transversais](#7-decisões-transversais)

---

## 1. Visão geral do projeto

O dataset GTSDB contém 900 fotografias de cenas de estrada (1360×800 px) com 1213 sinais de trânsito anotados em 43 classes. Estas 43 classes são agrupadas em 4 super-classes: Prohibitory, Danger, Mandatory e Other.

O trabalho está dividido em 4 tarefas progressivas que usam os mesmos dados mas formulam problemas diferentes:

| Task | Problema | Input | Output |
|------|----------|-------|--------|
| T1 | Classificação multi-class | Patch 96×96 (1 sinal) | 1 de 4 super-classes |
| T2 | Classificação multi-class | Patch 96×96 (1 sinal) | 1 de 43 classes finas |
| T3 | Classificação multi-label | Imagem completa 1360×800 | Conjunto de super-classes presentes |
| T4 | Deteção de objetos | Imagem completa 1360×800 | Bounding boxes + classe de cada sinal |

---

## 2. Pré-processamento

### 2.1 Conversão PPM → PNG

```python
for ppm in ppm_files:
    png = ppm.with_suffix('.png')
    if not png.exists():
        Image.open(ppm).save(png)
```

**Porquê:** O TensorFlow (`tf.io.read_file` + `tf.image.decode_image`) não suporta o formato PPM. A Pillow converte para PNG sem qualquer perda de qualidade (PNG é lossless). O `if not png.exists()` torna a célula idempotente — pode correr várias vezes sem reconverter.

### 2.2 Leitura do gt.txt

```python
gt = pd.read_csv(DATA_DIR / 'gt.txt', sep=';',
    names=['filename','x1','y1','x2','y2','class_id'],
    skipinitialspace=True)
gt['filename'] = gt['filename'].str.strip().str.replace('.ppm', '.png', regex=False)
```

**Porquê:** O `gt.txt` tem separador `;` com espaços em torno dele — por isso o `skipinitialspace=True` e o `.str.strip()`. Substituímos `.ppm` por `.png` para que os nomes de ficheiro no DataFrame correspondam aos ficheiros convertidos.

### 2.3 Mapeamento super-classes

```python
_MAP = [0,0,0,0,0,0,3,0,0,0,0, 1,3,3,3,0,0,3,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1, 3,2,2,2,2,2,2,2,2,3,3]
CLASS_TO_SUPERCLASS = np.array(_MAP)
```

**Porquê:** As 43 classes não estão distribuídas em intervalos contíguos por super-classe (por exemplo, a classe 6 é "Other" mas a 7 volta a ser "Prohibitory"). O array de 43 posições funciona como uma lookup table: `CLASS_TO_SUPERCLASS[class_id]` devolve a super-classe em O(1). Este mapeamento foi verificado contra o `ReadMe.txt` do dataset.

### 2.4 Extração de patches 96×96

```python
crop = img[int(row.y1):int(row.y2), int(row.x1):int(row.x2)]
patch = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
cv2.imwrite(str(PATCH_DIR / f'{basename}_{patch_index}.png'), patch)
```

**Porquê:** As Tasks 1 e 2 classificam sinais individuais, não cenas. Precisamos de recortar cada sinal usando as coordenadas do `gt.txt`. O tamanho 96×96 é razoável: grande o suficiente para preservar detalhes (formas, números de velocidade), pequeno o suficiente para treinar rapidamente.

`cv2.INTER_AREA` é a melhor interpolação para reduzir tamanho (downscale) — preserva mais informação do que `INTER_LINEAR` ou `INTER_NEAREST` ao comprimir pixels.

A convenção de nomes `{basename}_{index}.png` (ex: `00042_1.png`, `00042_2.png`) é exigida pelo enunciado.

### 2.5 Split 70/15/15

```python
train_df, temp_df = train_test_split(patches_df, test_size=0.30,
    stratify=patches_df['superclass'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50,
    stratify=temp_df['superclass'], random_state=SEED)
```

**Porquê `stratify`:** As 4 super-classes estão desequilibradas (Prohibitory: 557, Mandatory: apenas 163). Sem estratificação, o set de teste poderia ter poucos exemplos de Mandatory, tornando a avaliação pouco fiável. O `stratify` garante que a proporção de cada classe é mantida nos 3 subsets.

**Porquê um único split reutilizado em todas as tarefas:** Para que as comparações entre modelos sejam justas — T1 e T2 são avaliados exatamente nos mesmos patches de teste.

---

## 3. Task 1 — CNN from Scratch

### 3.1 Arquitetura

```
Input(96,96,3)
→ Conv2D(32, 3×3) + BatchNorm + ReLU + MaxPool(2×2)   → feature map: 48×48
→ Conv2D(64, 3×3) + BatchNorm + ReLU + MaxPool(2×2)   → feature map: 24×24
→ Conv2D(128, 3×3) + BatchNorm + ReLU + MaxPool(2×2)  → feature map: 12×12
→ Flatten → Dense(256) + ReLU + Dropout(0.5)
→ Dense(4) + Softmax
```

**Porquê 3 blocos convolucionais:** Após 3 max-poolings de 2×2, um input de 96×96 fica em 12×12. É uma resolução boa para o Flatten — nem demasiado grande (memória e overfitting) nem demasiado pequeno (perda de informação espacial).

**Porquê filtros 32→64→128:** Padrão de duplicar os filtros a cada bloco. As primeiras camadas detetam features simples (arestas, cores) — não precisam de muitos filtros. As camadas profundas combinam essas features em padrões mais complexos (forma do sinal) — precisam de mais capacidade.

**Porquê BatchNormalization:** Normaliza os activations entre camadas, o que permite learning rates mais altas sem instabilidade e reduz a dependência da inicialização dos pesos. Também tem um ligeiro efeito de regularização.

**Porquê Dropout(0.5):** Com apenas ~849 patches de treino (70% de 1213), o risco de overfitting é elevado. O Dropout desativa aleatoriamente 50% dos neurónios durante o treino, forçando o modelo a aprender representações redundantes em vez de memorizar os dados.

**Porquê Softmax na saída:** T1 é um problema **multi-class**: cada patch pertence a exatamente uma super-classe. O Softmax garante que as probabilidades das 4 classes somam 1 — matematicamente correto para classes mutuamente exclusivas.

### 3.2 Augmentation

```python
RandomFlip('horizontal')   # sinais podem aparecer dos dois lados da estrada
RandomRotation(0.05)       # ±5° — câmara pode não estar perfeitamente nivelada
RandomZoom(0.1)            # ±10% — variação de distância ao sinal
```

**Porquê NÃO usar `RandomFlip('vertical')` ou rotações grandes:** Sinais de trânsito têm orientação vertical definida — um sinal de STOP de cabeça para baixo não é um STOP. Rotações de 90° ou 180° criariam exemplos completamente irrealistas.

**Porquê aplicar augmentation apenas no treino:** A augmentation diversifica artificialmente os dados de treino. Na validação e teste queremos avaliar o modelo como vai ser usado na realidade — sem transformações artificiais.

### 3.3 Treino e callbacks

```python
tf.keras.optimizers.Adam(1e-3)
loss='sparse_categorical_crossentropy'
EarlyStopping(patience=10, restore_best_weights=True)
ReduceLROnPlateau(factor=0.5, patience=5)
```

**Porquê Adam:** Adapta a learning rate por parâmetro. Em datasets pequenos converge mais rápido e de forma mais estável que SGD puro.

**Porquê `sparse_categorical_crossentropy`:** Os labels são inteiros (0,1,2,3), não one-hot. A versão "sparse" aceita inteiros diretamente, evitando conversão desnecessária.

**Porquê `EarlyStopping` com `restore_best_weights=True`:** Para o treino quando a val_accuracy para de melhorar e repõe os pesos do melhor checkpoint, evitando usar um modelo que fez overfit nas últimas épocas.

**Porquê `ReduceLROnPlateau`:** Se a val_loss parar de melhorar por 5 épocas, reduz a lr a metade. Permite "afinar" em mínimos locais onde a lr original era demasiado grande para convergir.

### 3.4 Avaliação

**Confusion matrix não normalizada:** Mostra contagens absolutas de erros, o que permite identificar padrões (ex: o modelo confunde muito Danger com Prohibitory porque ambos têm formas circulares).

**ROC curves por classe:** Para cada classe, calcula a curva ROC tratando o problema como binário (essa classe vs. todas as outras). O AUC indica quão bem o modelo separa cada classe do resto, independentemente do threshold.

**8 corretos + 8 errados:** Visualização qualitativa — fundamental para perceber se os erros são "razoáveis" (sinais pequenos, ambíguos) ou sistemáticos (o modelo nunca acerta numa classe específica).

---

## 4. Task 2 — Transfer Learning (43 classes)

### 4.1 Porquê MobileNetV2

MobileNetV2 usa **depthwise separable convolutions**: em vez de uma convolução 3×3 normal, faz uma convolução 3×3 por canal (depthwise) seguida de uma 1×1 para combinar canais (pointwise). O número de operações cai drasticamente, tornando o modelo muito mais leve.

Outras opções consideradas:
- **VGG16**: muito mais pesado (138M parâmetros), criado para 224×224 — não justifica o custo extra para patches 96×96.
- **EfficientNetV2**: melhor para imagens de alta resolução, mais complexo de configurar.
- **ResNet50**: boa opção, mas mais parâmetros que MobileNetV2 sem ganho proporcional para este dataset.

### 4.2 Estratégia dois fases

**O problema do catastrophic forgetting:**

Os pesos do MobileNetV2 foram treinados no ImageNet com 1.4M imagens. Se adicionarmos um head aleatório e treinarmos tudo ao mesmo tempo com lr=1e-3, os gradientes grandes do head (que ainda não sabe nada) vão destruir esses pesos. Isto chama-se catastrophic forgetting.

**Solução:**
```
Fase 1: backbone.trainable = False → lr = 1e-3 (aprende o head)
Fase 2: backbone.layers[:-20].trainable = False → lr = 1e-5 (fine-tune)
```

**Porquê descongelar as últimas 20 camadas (não todas):**

As primeiras camadas de uma CNN aprendem features genéricas (arestas, texturas) que são iguais em qualquer dataset. As últimas camadas aprendem features específicas ao domínio. Para sinais de trânsito, as features genéricas do ImageNet são úteis — só precisamos de adaptar as últimas camadas.

**Porquê lr=1e-5 na fase 2:** Um valor 100× mais baixo que a fase 1, para fazer ajustes mínimos nos pesos do backbone sem os destruir.

### 4.3 Class imbalance

```python
cw = compute_class_weight('balanced', classes=np.arange(43), y=train_df['class_id'].values)
```

**O problema:** Algumas classes têm 50+ exemplos de treino, outras têm menos de 5. Sem class weights, o modelo ignora as classes raras e maximiza accuracy nas frequentes.

**A solução `balanced`:** Calcula `weight_i = total / (n_classes × count_i)`. Classes raras recebem pesos mais altos — o modelo é mais penalizado por errar nessas classes.

**Porquê não oversample:** Criar patches duplicados pode levar a overfitting nas classes raras. Os class weights são mais elegantes e sem custo computacional.

### 4.4 Head de classificação

```python
GlobalAveragePooling2D()
Dense(256) + Dropout(0.3)
Dense(43) + Softmax
```

**Porquê `GlobalAveragePooling2D` em vez de `Flatten`:** O Flatten de um feature map 3×3×1280 produziria 11520 dimensões — grande demais, propenso a overfitting. O GAP faz a média espacial por canal, produzindo 1280 dimensões. É mais robusto a variações de posição do objeto.

**Porquê Dropout(0.3) e não 0.5:** Com backbone pré-treinado, há muito menos tendência a overfit. Um Dropout mais suave é suficiente.

---

## 5. Task 3 — Multi-label Classification

### 5.1 A diferença fundamental

Em T1 e T2, cada patch pertence a **exatamente uma** classe. Uma cena completa pode conter **zero, um, ou vários** sinais de tipos diferentes em simultâneo.

| Aspeto | Multi-class (T1/T2) | Multi-label (T3) |
|--------|---------------------|------------------|
| Output layer | `Dense(N) + Softmax` | `Dense(N) + Sigmoid` |
| Loss | `sparse_categorical_crossentropy` | `binary_crossentropy` |
| Predição | `argmax(outputs)` | `outputs >= threshold` |
| Probabilidades | Somam 1 | Independentes entre si |

**Porquê Sigmoid:** O Softmax forçaria as probabilidades a somar 1 — se Prohibitory sobe, as outras baixam. Mas numa cena com Prohibitory e Danger, queremos AMBAS altas. O Sigmoid produz uma probabilidade independente por saída.

**Porquê `binary_crossentropy`:** O problema é tratado como 4 classificações binárias independentes: "há Prohibitory? Sim/Não", etc. A binary cross-entropy é a loss correta para classificação binária.

### 5.2 Augmentation diferente dos patches

```python
RandomFlip('horizontal')   # válido para cenas de estrada
RandomBrightness(0.1)      # variação de iluminação (manhã, tarde, nublado)
RandomContrast(0.1)        # variação de contraste
# SEM RandomRotation — o horizonte ficaria inclinado
# SEM RandomZoom — poderia cortar sinais nas bordas
```

### 5.3 Análise de thresholds (0.3, 0.5, 0.7)

**Porquê testar 3 thresholds:** O threshold 0.5 é o padrão, mas nem sempre é o ótimo. Threshold baixo → mais Recall (deteta mais, incluindo com baixa confiança) mas menos Precision. Threshold alto → o contrário. Para segurança rodoviária, Recall alto costuma ser prioritário.

### 5.4 Resolução 224×224 (não a original 1360×800)

**Porquê:** O MobileNetV2 foi desenhado para 224×224 — os seus pesos ImageNet são ótimos para esta resolução. Usar 1360×800 exigiria batches de 1-2 imagens (instabilidade de treino) e muito mais memória GPU. Como T3 só classifica quais super-classes estão presentes (não onde), a perda de detalhe é aceitável.

---

## 6. Task 4 — Object Detection com YOLOv8

### 6.1 Porquê YOLOv8 e não RetinaNet

- **Single-pass:** processa a imagem numa única passagem pela rede.
- **Anchor-free:** não requer definição manual de anchors, simplificando o pipeline.
- **Melhor suporte em KerasCV** com exemplos mais abundantes.

### 6.2 Problem Adaptation — análise das bounding boxes

```python
gt['box_w_640'] = (gt['x2'] - gt['x1']) * scale_x
gt['box_h_640'] = (gt['y2'] - gt['y1']) * scale_y
```

**Porquê obrigatória:** Os sinais do GTSDB variam de 15×15 px a 250×250 px. Antes de configurar o modelo, precisamos de saber a distribuição real dos tamanhos para verificar que o Feature Pyramid Network (FPN) cobre bem todas as escalas, e para comparar com os anchors COCO (desenhados para objetos maiores).

### 6.3 Negative samples (159 imagens sem sinais)

**Porquê incluir todas:** Se o modelo nunca vir imagens de fundo puro durante o treino, aprende a detetar sinais mas não a inibir deteções falsas em cenas limpas — taxa elevada de falsos positivos. Incluindo todas as 159 imagens com bboxes vazias, o modelo aprende que a maioria dos pixels de estrada é background.

### 6.4 Augmentation bbox-aware

```python
keras_cv.layers.RandomFlip(mode='horizontal', bounding_box_format='xyxy')
```

**Porquê usar KerasCV e não tf.keras:** Quando augmentamos imagens de deteção, as bounding boxes têm de ser transformadas junto com a imagem. As camadas KerasCV são "bbox-aware" — transformam imagem e boxes de forma consistente. Sem isto, o modelo aprenderia que os sinais estão no lugar errado após o flip.

**Verificação obrigatória:** Visualizamos um batch com as boxes desenhadas sobre as imagens augmentadas antes de iniciar o treino. Se as boxes não alinharem com os sinais, há um bug no pipeline.

### 6.5 Formato das bounding boxes

```python
{'boxes': tf.constant([[x1,y1,x2,y2]], dtype=tf.float32),
 'classes': tf.constant([superclass_id], dtype=tf.float32)}

# Para imagens sem sinais:
{'boxes': tf.zeros([0,4]), 'classes': tf.zeros([0])}
```

**Porquê `ragged_batch`:** Imagens diferentes têm 0 a 6 bounding boxes. O `batch` normal exige tensores da mesma shape. O `ragged_batch` cria Ragged Tensors que suportam dimensões de tamanho variável — essencial para deteção de objetos.

### 6.6 Loss de deteção

**CIoU para as boxes:** Mede a sobreposição entre a box predita e a ground truth, penalizando também a diferença de aspect ratio e distância entre centros. Superior ao MSE simples sobre coordenadas porque é invariante à escala.

**Binary cross-entropy para as classes:** Cada super-classe é tratada de forma independente — o mesmo raciocínio do T3.

### 6.7 Métricas de deteção

**mAP@0.5:** Média da Average Precision com threshold de IoU de 0.5. Uma predição conta como correta se a box predita sobrepõe pelo menos 50% da ground truth e a classe está certa.

**mAP@0.5:0.95:** Média do mAP para thresholds de IoU entre 0.5 e 0.95 (passo 0.05). Métrica mais exigente — penaliza boxes imprecisas mesmo que a classe esteja certa. Standard COCO.

**False positive rate nas 159 imagens sign-free:** Fração de imagens sem sinais onde o modelo produz pelo menos uma deteção. Importante para segurança — um sistema de condução autónoma não deve "ver" sinais onde não existem.

---

## 7. Decisões transversais

### 7.1 Seed global

```python
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
```

Garante reproducibilidade: o mesmo split, os mesmos batches aleatórios, e os mesmos pesos iniciais em cada execução. Fundamental para poder comparar resultados de diferentes runs.

### 7.2 Normalização de imagens

Para a CNN do zero (T1): dividir por 255 → escala [0, 1].

Para MobileNetV2 (T2, T3, T4):
```python
tf.keras.applications.mobilenet_v2.preprocess_input(inputs)  # escala [-1, 1]
```

**Porquê diferente:** O MobileNetV2 foi treinado com a sua própria função de pré-processamento que escala para [-1, 1]. Usar [0, 1] em vez disso alimenta o backbone com uma distribuição diferente da que foi usada no ImageNet — os pesos pré-treinados deixam de ser válidos.

### 7.3 `prefetch` e `num_parallel_calls=AUTOTUNE`

```python
ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
```

Sem estas otimizações, a GPU fica à espera que a CPU carregue e processe as imagens para o próximo batch. Com `AUTOTUNE`, o TF determina automaticamente o paralelismo e faz prefetch assíncrono — a CPU prepara o batch N+1 enquanto a GPU treina com o batch N. Reduz significativamente o tempo de treino em datasets de imagens.

### 7.4 Caminhos relativos

```python
DATA_DIR  = Path('images/FullIJCNN2013')
PATCH_DIR = Path('patches')
```

O notebook é entregue sem as imagens (são demasiado grandes). Caminhos relativos garantem que funciona em qualquer máquina com a estrutura de pastas correta. Caminhos absolutos (`/Users/btavr/...`) só funcionariam no computador do autor.

### 7.5 Monitorizar `val_accuracy` e não `val_loss`

Para T1 e T2 monitoramos `val_accuracy` no `EarlyStopping` e `ModelCheckpoint`.

**Porquê:** Em datasets pequenos, a val_loss pode oscilar mesmo quando a val_accuracy está estável. Monitorar a accuracy dá um sinal mais direto de "o modelo está a classificar melhor?" — que é o objetivo final.
