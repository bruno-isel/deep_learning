# Titanic Dataset — Worksheet 02

Notebook: `Titanic WorkSheet.ipynb` | Dataset: `titanic.csv` (1309 passageiros, 14 colunas)

Prever sobrevivência (`survived`) com regressão logística, comparando duas codificações da feature `embarked`.

---

## Part A — Load Data

O CSV é separado por tabulações e usa vírgula decimal (`sep='\t', decimal=','`).

```python
df = pd.read_csv("titanic.csv", sep="\t", decimal=",")
```

Shape: `(1309, 14)`.

## Part B — Passengers from Spain

Filtro com `str.contains("spain", case=False, na=False)` sobre `home.dest`.  
`na=False` é necessário porque há valores `NaN` nessa coluna que causariam erro.

Resultado: 7 passageiros com destino/origem em Espanha.

## Part C — Survival Rates

Calculadas com `.mean()` sobre a coluna `survived` (0/1):

| Grupo | Taxa |
|-------|------|
| Mulheres | ~72.7% |
| Homens | ~19.1% |
| 1ª classe | ~61.9% |

## Part D — Age Histogram

`plt.hist` sobreposto (com `alpha=0.5`) para sobreviventes vs mortos.  
`.dropna()` remove os NaN de `age` antes de plotar.  
Guardado em `fig_age_hist.png`.

## Part E — Imputations → `df_imp`

| Coluna | Estratégia | Razão |
|--------|-----------|-------|
| `embarked` | Substituir NaN por `"C"` | Cherbourg é o porto mais frequente |
| `fare` | Mediana de `pclass==3` | Os NaN são de 3ª classe; mediana é robusta a outliers |
| `age` | Mediana por grupo `(sex, pclass)` via `groupby().transform("median")` | A idade típica varia muito por género e classe |

Após imputação: 0 valores em falta nas três colunas.

## Part F — Feature Matrix `Xa`

Colunas: `[sex, age, sibsp, parch, fare, embarked_num]`

- `sex`: male=0, female=1
- `embarked_num`: C→0, Q→1, S→2 (encoding ordinal — impõe ordem artificial entre portos)

Shape: `(1309, 6)`.

## Part G — Feature Matrix `Xb`

Colunas: `[sex, age, sibsp, parch, fare, emb_Q, emb_S]`

- C é a baseline (quando `emb_Q=0` e `emb_S=0`)
- **One-hot encoding** correcto: não impõe ordem artificial entre os portos

Shape: `(1309, 7)`.

## Part H — Logistic Regression + ROC + Confusion Matrix

- Split estratificado: `test_size=0.2, random_state=42, stratify=y`
- Standardização manual (média e std calculadas só no train — evita data leakage sem usar Pipeline)
- `LogisticRegression(max_iter=2000)` nos dados normalizados

| Modelo | AUC |
|--------|-----|
| `model_Xa` | ~0.855 |
| `model_Xb` | ~0.856 |

Xb tem AUC ligeiramente superior porque a codificação one-hot é matematicamente mais correcta para variáveis nominais.  
Figuras: `fig_roc_compare.png`, `fig_cm_Xb.png`.

## Part I — Feature Weights

Top features por valor absoluto do coeficiente (dados standardizados — coeficientes comparáveis):

| Rank | Xa | Xb |
|------|----|----|
| 1 | **sex** (1.29) | **sex** (1.32) |
| 2 | **age** (0.54) | **age** (0.50) |
| 3 | embarked_num (-0.23) | emb_S (-0.32) |

`sex` é de longe a feature mais discriminante — confirma a política "mulheres e crianças primeiro".  
`age` é a segunda mais importante. O porto de embarque tem impacto menor mas real.
