# Model Card - ChurnClassifier

## Modèle et finalité

`ChurnClassifier` est une régression logistique destinée à estimer la probabilité de résiliation d'un abonnement télécom. Elle sert à prioriser une analyse humaine ou une campagne de rétention, et non à prendre seule une décision affectant un client.

## Données

- Source: IBM Telco Customer Churn, 7 043 observations.
- Cible positive: `Churn = Yes` (26,5 % environ).
- Entraînement/test: split stratifié 80/20, graine 42.
- Variables: 3 numériques et 16 catégorielles; `customerID` est exclue.
- Valeurs manquantes de `TotalCharges`: imputées par médiane dans les plis d'entraînement.

## Sélection et résultats

La pipeline complète est optimisée par `GridSearchCV`, avec 5 plis stratifiés et le ROC-AUC comme objectif. La configuration retenue est `C=10`, `solver=liblinear`, `penalty=l2`, sans pondération de classes.

| Mesure | Valeur test |
|---|---:|
| ROC-AUC | 0,8411 |
| Average precision | 0,6281 |
| Accuracy | 0,8048 |
| Balanced accuracy | 0,7263 |
| F1-score | 0,6032 |

## Limites et risques

- Le dataset est un échantillon pédagogique ancien; il ne prouve pas la performance sur une clientèle réelle actuelle.
- Le seuil de 0,50 n'est pas optimisé selon le coût métier des faux négatifs/faux positifs.
- L'accuracy masque partiellement le déséquilibre de classe; ROC-AUC, average precision, rappel et matrice de confusion doivent être lus ensemble.
- Les performances par sous-groupes et la calibration doivent être auditées avant production.
- Toute mise en production exige monitoring de dérive, réentraînement gouverné et validation humaine.

## Traçabilité

Le modèle, la signature, l'exemple d'entrée, les paramètres, métriques, résultats de validation croisée et graphiques sont tracés dans MLflow. La version validée est enregistrée sous `ChurnClassifier`, alias `staging` et stage historique `Staging` lorsque disponible.
