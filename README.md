# Churn Classifier with Pipelines + MLflow

[![CI](https://github.com/ezzatsd/churn-classifier-mlflow/actions/workflows/ci.yml/badge.svg)](https://github.com/ezzatsd/churn-classifier-mlflow/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Projet MLOps reproductible de classification du churn client sur le jeu de données public **IBM Telco Customer Churn**. Le projet assemble le prétraitement et le modèle dans une seule pipeline anti-fuite, optimise les hyperparamètres par validation croisée stratifiée et trace l'expérience complète dans MLflow.

## Points forts

- `Pipeline` + `ColumnTransformer`: imputation, standardisation et one-hot encoding appris exclusivement sur les plis d'entraînement.
- `GridSearchCV` avec `StratifiedKFold` à 5 plis et optimisation du ROC-AUC.
- MLflow: configuration, hyperparamètres, métriques, artefacts, modèle signé et Model Registry.
- Évaluation de holdout: ROC, précision-rappel, matrice de confusion et prédictions ligne à ligne.
- CLI, YAML, Makefile, tests, lint, Docker et CI GitHub Actions.
- Reproductibilité: split stratifié et graines aléatoires fixes.
- Sécurité: audit de dépendances, checksum du dataset, conteneur non-root et actions épinglées.

## Architecture

```text
.
├── configs/config.yaml
├── data/                       # données locales ignorées par Git
├── src/
│   ├── pipeline.py             # prétraitement + estimateur
│   ├── train.py                # tuning, tracking et registry
│   ├── evaluate.py             # évaluation holdout et artefacts
│   ├── predict.py              # inférence CSV batch
│   ├── download_data.py        # téléchargement IBM reproductible
│   └── utils.py
├── tests/
├── .github/workflows/ci.yml
├── Dockerfile
├── Makefile
└── requirements.txt
```

## Demarrage rapide

```bash
make init
make data
make test
make lint
make audit
make train
make evaluate
make mlflow-ui
```

Ouvrir ensuite <http://127.0.0.1:5000>. Le registre contient `ChurnClassifier`; sa meilleure version reçoit l'alias `staging` et, lorsque la version de MLflow le permet, le stage historique `Staging`.

## Configuration

Tous les chemins, colonnes, paramètres de split, grille, validation croisée et réglages MLflow sont centralisés dans [`configs/config.yaml`](configs/config.yaml). Les variables de [`.env.example`](.env.example) permettent de remplacer l'URI de tracking et le nom d'expérience sans modifier le code.

La colonne identifiante `customerID` est volontairement exclue. Les 11 valeurs blanches de `TotalCharges` sont converties en valeurs manquantes, puis imputées dans la pipeline. Cela garantit que la médiane est apprise uniquement sur l'entraînement.

## Inference batch

Après l'entraînement:

```bash
python -m src.predict \
  --config configs/config.yaml \
  --input data/raw.csv \
  --output artifacts/batch_predictions.csv
```

Le CSV produit conserve les colonnes d'entrée et ajoute `churn_probability` et `predicted_churn`.

## Docker

```bash
docker build -t churn-classifier .
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" churn-classifier
```

## Résultats vérifiés

Exécution du 7 septembre 2026 sur le split déterministe (`random_state=42`):

| Mesure | Validation croisée | Test final (1 409 clients) |
|---|---:|---:|
| ROC-AUC | **0,8463** | **0,8411** |
| Average precision | - | 0,6281 |
| Accuracy | - | 0,8048 |
| Balanced accuracy | - | 0,7263 |
| Précision | - | 0,6552 |
| Rappel | - | 0,5588 |
| F1-score | - | 0,6032 |

Meilleure configuration: `C=10`, `solver=liblinear`, `penalty=l2`, sans pondération de classes. Matrice de confusion: TN=925, FP=110, FN=165, TP=209. Le registre MLflow contient la version 1 de `ChurnClassifier` en `Staging`.

Qualité vérifiée localement: **8 tests réussis** et **0 erreur Ruff**.

## Donnees et ethique

Source: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d). Les données ne sont pas versionnées dans Git; `make data` télécharge 7 043 lignes depuis le dépôt IBM et exige le SHA-256 attendu. Le modèle aide à prioriser une analyse de rétention: il ne doit pas déclencher seul une décision affectant un client. Il faut surveiller dérive, calibration et écarts de performance entre sous-groupes avant toute mise en production. Voir la [model card](docs/model_card.md) et la [politique de sécurité](SECURITY.md).

## Licence

Code publié sous licence MIT. Le jeu de données reste soumis aux conditions de sa source.
