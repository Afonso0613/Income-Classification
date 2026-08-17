# Classificação de Income (Adult / Census Income Dataset)

Projeto académico de Machine Learning que compara vários algoritmos de classificação supervisionada — e uma abordagem não supervisionada — na previsão de se o rendimento anual de um indivíduo ultrapassa os 50 mil dólares, a partir de dados sociodemográficos.

## Sobre o projeto

O dataset (~32.500 registos) contém atributos como idade, escolaridade, ocupação, estado civil, raça, género, horas de trabalho semanais e país de origem, com a variável alvo `income` (`<=50K` ou `>50K`). O objetivo é tratar e explorar estes dados e treinar/comparar múltiplos modelos de classificação para prever a classe de rendimento.

O trabalho está dividido em três fases: **tratamento de dados**, **análise exploratória** e **modelação** (7 algoritmos supervisionados + 1 abordagem não supervisionada), todos avaliados por validação cruzada estratificada (5-fold).

## Estrutura do projeto

```
.
├── DATASETS/
│   ├── salario.csv                 # Dataset original (bruto)
│   └── salario_tratamento.csv      # Dataset tratado (output de Tratamento_de_Dados.ipynb)
└── NOTEBOOKS/
    ├── Tratamento_de_Dados.ipynb   # Limpeza e preparação dos dados
    ├── Análise_de_Dados.ipynb      # Análise exploratória / visualizações
    ├── Regressao_logistica.ipynb   # Modelo: Regressão Logística
    ├── SVM.ipynb                   # Modelo: Support Vector Machine
    ├── DecisionTree.ipynb          # Modelo: Decision Tree
    ├── RandomForest.ipynb          # Modelo: Random Forest
    ├── GradientBoosted.ipynb       # Modelo: Gradient Boosting
    ├── XGBoost.ipynb               # Modelo: XGBoost
    ├── RedesNeuronais.ipynb        # Modelo: Rede Neuronal (Keras/TensorFlow)
    └── NaoSupervisionada.ipynb     # Clustering K-Means (não supervisionado)
```

> Os notebooks de modelação carregam sempre `../DATASETS/salario_tratamento.csv`, pelo que assumem esta estrutura de pastas (`DATASETS/` ao lado de `NOTEBOOKS/`).

## Datasets

- **`salario.csv`** — dataset original, baseado no *Adult / Census Income* dataset. Colunas: `age`, `workclass`, `fnlwgt`, `education`, `education.num`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `capital.gain`, `capital.loss`, `hours.per.week`, `native.country`, `income`. Valores em falta representados por `"?"`.
- **`salario_tratamento.csv`** — dataset já tratado (output de `Tratamento_de_Dados.ipynb`), pronto a ser consumido pelos notebooks de modelação.

## 1. Tratamento de Dados (`Tratamento_de_Dados.ipynb`)

- Substituição dos valores `"?"` por *missing values* e imputação pela **moda** nas colunas `occupation`, `workclass` e `native.country`.
- Criação da coluna `capital.diff` = `capital.gain` − `capital.loss`, e remoção de `capital.gain`/`capital.loss`.
- Remoção das colunas `education.num` e `fnlwgt` (redundantes/pouco informativas).
- *Binning* da idade em faixas etárias (`age.group`): `17-25`, `26-35`, `36-45`, `46-55`, `56-65`, `66-90`; remoção da coluna `age` original.
- Exportação do resultado para `salario_tratamento.csv`.

## 2. Análise Exploratória (`Análise_de_Dados.ipynb`)

Estatísticas descritivas e visualizações sobre o dataset original: número total de entradas, histogramas (idade, horas semanais), gráficos de barras (classe de trabalho, educação, estado civil, ocupação, relação familiar, raça) e *pie charts* (sexo, distribuição de `income`).

## 3. Modelação — pré-processamento comum

Todos os notebooks de modelação seguem o mesmo pipeline antes de treinar o modelo:

1. Carregam `salario_tratamento.csv` e separam `x` (atributos) e `y` (`income`).
2. **Label Encoding** nas colunas categóricas: `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`, `age.group`.
3. **Normalização** (`StandardScaler`) nas colunas numéricas: `capital.diff`, `hours.per.week`.
4. Treino e avaliação por **validação cruzada estratificada** (`StratifiedKFold`, 5 folds), com métricas *accuracy*, *precision*, *recall* e matriz de confusão.

## 4. Modelos Supervisionados e Resultados

| Notebook | Modelo | Accuracy | Precision | Recall |
|---|---|---|---|---|
| `XGBoost.ipynb` | XGBoost | **0.872** | 0.774 | 0.664 |
| `GradientBoosted.ipynb` | Gradient Boosting | 0.864 | 0.789 | 0.593 |
| `RandomForest.ipynb` | Random Forest | 0.851 | 0.719 | 0.626 |
| `RedesNeuronais.ipynb` | Rede Neuronal (Keras) | 0.843 | 0.713 | 0.594 |
| `DecisionTree.ipynb` | Decision Tree | 0.827 | 0.653 | 0.605 |
| `SVM.ipynb` | SVM (kernel RBF) | 0.792 | 0.986 | 0.140 |
| `Regressao_logistica.ipynb` | Regressão Logística | 0.791 | 0.666 | 0.263 |

**XGBoost** foi o modelo com melhor desempenho global (melhor equilíbrio entre accuracy, precision e recall). O **SVM** destaca-se por precision quase perfeita, mas com recall muito baixo — sinal de forte tendência para prever a classe maioritária (`<=50K`).

A rede neuronal (`RedesNeuronais.ipynb`) usa uma arquitetura simples *feedforward* (Dense 128 → Dense 64 → Dense 1, ativações ReLU/sigmoid, otimizador Adam, *loss* binary cross-entropy), treinada 10 épocas por fold.

## 5. Análise Não Supervisionada (`NaoSupervisionada.ipynb`)

Aplicação de **K-Means** (3 clusters) sobre os dados tratados e normalizados, avaliado com *silhouette score* e inércia, para explorar se emergem agrupamentos naturais nos dados sem recorrer à variável `income`.

## Como executar

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost tensorflow

# Ordem recomendada:
# 1. Tratamento_de_Dados.ipynb      (gera salario_tratamento.csv)
# 2. Análise_de_Dados.ipynb          (opcional, análise exploratória)
# 3. Qualquer um dos notebooks de modelação (Regressao_logistica, SVM, DecisionTree,
#    RandomForest, GradientBoosted, XGBoost, RedesNeuronais, NaoSupervisionada)
```

## Tecnologias

Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras · Matplotlib · Seaborn
