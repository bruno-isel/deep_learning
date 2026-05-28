# Scikit-Learn Intro — Worksheet 01

Notebook: `Introduction to Scikit-Learn WorkSheet.ipynb` | Dataset: Wisconsin Breast Cancer (Diagnostic)

Classificação binária: tumor **maligno (0)** vs **benigno (1)** — 569 amostras, 30 features.

---

## Part 1 — Load Dataset

`load_breast_cancer()` do scikit-learn devolve features em `X` e labels em `y`.  
`np.unique(y, return_counts=True)` confirma o desequilíbrio: 212 malignos vs 357 benignos.

## Part 2 — Stratified Train/Test Split

`train_test_split(..., stratify=y)` garante que a proporção de classes é igual no train e no test.  
Sem estratificação, um split aleatório pode concentrar mais exemplos de uma classe num lado.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## Part 3 — Pipeline + Logistic Regression

O `Pipeline` encadeia pré-processamento e modelo num único objeto — evita data leakage porque o `StandardScaler` só vê os dados de treino (fit no train, transform em ambos).

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(max_iter=2000))
])
```

`test_accuracy ≈ 0.982` — os nomes dos steps (`"scaler"`, `"clf"`) são obrigatórios para o grader e para o `GridSearchCV` (prefixo `clf__`).

## Part 4 — Confusion Matrix + Metrics

|  | Previsto 0 | Previsto 1 |
|--|--|--|
| **Real 0** | 41 (TN) | 1 (FP) |
| **Real 1** | 1 (FN) | 71 (TP) |

- **Precision** = TP / (TP + FP) — quantos dos previstos positivos são realmente positivos
- **Recall** = TP / (TP + FN) — quantos dos positivos reais foram detectados
- **F1** = média harmónica de precision e recall

Resultado: precision = recall = F1 ≈ **0.986**.

## Part 5 — Decision Threshold Adjustment

Por defeito, `predict` usa threshold = 0.5. Em diagnóstico médico, **recall alto** é prioritário (minimizar falsos negativos = tumores não detectados).

```python
probs_pos = pipe.predict_proba(X_test)[:, 1]
tau = 0.30
y_pred_tau = (probs_pos >= tau).astype(int)
```

Com `tau = 0.30`: recall sobe para **1.0** (sem FN), mas aumenta ligeiramente os FP (2 em vez de 1).

## Part 6 — ROC Curve + AUC

A curva ROC mostra o trade-off TPR/FPR para todos os thresholds possíveis.  
`AUC ≈ 0.995` — praticamente perfeito para este dataset.

```python
fpr, tpr, _ = roc_curve(y_test, probs_pos)
roc_auc = auc(fpr, tpr)
```

## Part 7 — Cross-Validation (ShuffleSplit)

`ShuffleSplit(n_splits=20, test_size=0.2)` repete o split 20 vezes com partições aleatórias diferentes, dando uma estimativa mais robusta da performance do que um único split.

`cv_mean ≈ 0.981`, `cv_std ≈ 0.012` — baixo desvio padrão indica modelo estável.

## Part 8 — GridSearchCV (Hyperparameter Search)

Pesquisa exaustiva sobre `C` (inverso da regularização L2):

```python
param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]}
grid = GridSearchCV(estimator=pipe, cv=5, scoring="f1", n_jobs=-1)
```

`best_params = {'clf__C': 0.1}` — regularização ligeiramente mais forte que o default (C=1) produz melhor F1 com CV=5.
