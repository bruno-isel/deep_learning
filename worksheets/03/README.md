# Object Detection with KerasCV — Worksheet 03

Notebook: `object_detection.ipynb` | Dataset: 10 COCO images in `coco_images/`

---

## Part 1 — Pre-trained YOLOv8 Inference (35 pts)

### 1.1 Load Images
`tf.image.resize_with_pad` redimensiona cada imagem para `(640, 640)` mantendo o aspect ratio e preenchendo com preto. As 10 imagens são empilhadas num tensor `(10, 640, 640, 3) float32`.

### 1.2 Visualise Images
Grid 5×2 com `matplotlib`, título = nome do ficheiro.

### 1.3 Load Model and Run Predictions
`keras_cv.models.YOLOV8Detector.from_preset('yolo_v8_m_pascalvoc')` carrega o modelo pré-treinado. `model.predict(images)` devolve um dict com `boxes`, `confidence`, `classes`.

### 1.4 Visualise Detections
Itera sobre as boxes de cada imagem e desenha rectângulos vermelhos para detecções com `confidence > 0.2`. Label = nome da classe + score.

### 1.5 Fewer Bounding Boxes
`set_nms(model, iou_thresh=0.3, conf_thresh=0.5)` — threshold de confiança alto descarta caixas fracas; IoU threshold baixo torna o NMS mais agressivo (suprime mais sobreposições).

### 1.6 More Bounding Boxes
`set_nms(model, iou_thresh=0.7, conf_thresh=0.05)` — threshold de confiança baixo aceita caixas fracas; IoU threshold alto permite mais sobreposição antes de suprimir.

### 1.7 Comparison (optional)
Comparação lado a lado de `det_few` vs `det_many` para a imagem 0.

---

## Part 2 — Intersection over Union (10 pts)

IoU = área da interseção / área da união. Threshold padrão: `IOU_THR = 0.50`.

- `PRED_A` tem grande sobreposição com `GT_BOX` → **TP**
- `PRED_B` não sobrepõe `GT_BOX` → **FP**

```
is_tp_A = iou_A >= IOU_THR   # True
is_tp_B = iou_B >= IOU_THR   # False
```

---

## Part 3 — Ground-Truth Annotations (20 pts)

### 3.1 Parse Annotations
- `gt_boxes` — lista de listas com coordenadas `[x1, y1, x2, y2]` como floats
- `gt_classes` — lista de listas com IDs inteiros (convertidos via `CMAP_INV`)

### 3.2 Per-Class GT Count Table
`defaultdict(int)` conta GT boxes por nome de classe em todas as 10 imagens. Apresentado como tabela pandas ordenada por contagem decrescente.

---

## Part 4 — Evaluation Pipeline (35 pts)

### 4.1 Build the tf.data Pipeline
`tf.ragged.constant` é necessário porque cada imagem tem número diferente de boxes. Pipeline:
```
from_tensor_slices → map(_load_image_bb) → ragged_batch(10) → map(_resize_image_bb) → cache
```
As bounding boxes são transformadas pelo `imResize` juntamente com as imagens.

### 4.2 Visualise GT vs Predicted
- **Verde** = ground truth
- **Laranja** = predições com `conf > 0.2`

### 4.3 COCO Evaluation Metrics
1. Modelo fresco (sem alterações de NMS) para avaliação justa
2. `pad_gt_to_fixed` converte RaggedTensors em densos (padding com `-1`) — necessário para `BoxCOCOMetrics`
3. `coco_metric.update_state(y_true, preds)` + `.result()` devolve mAP em vários thresholds de IoU

### 4.4 AP@0.50 for `person`

Protocolo Pascal VOC 2010+:

1. Correr com `EVAL_THR = 0.05` para capturar a curva P/R completa
2. Recolher todas as predições `person` nas 10 imagens; ordenar por confiança decrescente
3. **Matching greedy**: para cada predição (por ordem de confiança), encontrar a GT box com maior IoU; se `IoU >= 0.50` e a GT ainda não foi emparelhada → **TP**, caso contrário → **FP**
4. Calcular precision e recall cumulativos
5. **Interpolação monotónica VOC**: `np.maximum.accumulate` da direita para a esquerda elimina o efeito de serra da curva
6. `AP = Σ (recall[i+1] − recall[i]) × precision_smooth[i+1]`
