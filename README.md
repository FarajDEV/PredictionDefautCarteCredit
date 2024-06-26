<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" height="128">
  <h2 align="center"><a href="https://www.linkedin.com/in/cheniki-faraj-%F0%9F%91%A8%E2%80%8D%F0%9F%92%BB-575a352b7/">LinkedIn</a></h2>
</p>

<br>

![](https://i.imgur.com/waxVImv.png)

# Prédiction de Défaut de Carte de Crédit avec AWS SageMaker Studio 🚀

## Projet N°3 🔄 🔐
#  🏛️✨PredictionDefautCarteCredit✨🏛️

Bienvenue dans notre projet de **Prédiction de Défaut de Carte de Crédit**. 
Ce projet utilise AWS SageMaker Studio pour développer, entraîner et déployer un modèle de machine learning capable de prédire les défauts de paiement des clients de cartes de crédit. 🌟

## Table des Matières
- [Description 📖](#description-)
- [Cahier des Charges 📋](#cahier-des-charges-)
- [Fonctionnalités ✨](#fonctionnalités-)
- [Structure du Projet 🏗️](#structure-du-projet-)
- [Réalisation 🛠️](#réalisation-)
- [Structure du Code 🧩](#structure-du-code-)
- [Difficultés Rencontrées 🚧](#difficultés-rencontrées-)
- [Améliorations Futures 🔮](#améliorations-futures-)
- [Frameworks et Technologies Utilisés 🛠️](#frameworks-et-technologies-utilisés-)
- [Auteurs et Contributions ✨](#auteurs-et-contributions-)
- [Références et Ressources 📚](#références-et-ressources-)

## Description 📖

### Contexte
Dans le secteur bancaire, l'évaluation du risque de crédit est essentielle pour réduire les pertes financières dues aux défauts de paiement. Les modèles de machine learning peuvent grandement améliorer la précision de ces évaluations en analysant des données complexes et volumineuses.

### Objectif
Le but de ce projet est de développer un modèle de machine learning utilisant **XGBoost** pour prédire la probabilité qu'un client de carte de crédit ne puisse pas honorer ses paiements. Nous utilisons AWS SageMaker Studio pour l'entraînement, l'optimisation et le déploiement de notre modèle.

## Cahier des Charges 📋

### Besoins Fonctionnels
- **Pré-traitement des données** pour améliorer la qualité des entrées du modèle.
- **Entraînement et optimisation** d'un modèle de machine learning performant.
- **Déploiement du modèle** sur AWS SageMaker pour une utilisation en production.
- **Évaluation continue** des performances du modèle et ajustements si nécessaire.

### Contraintes Techniques
- Utilisation de **AWS SageMaker Studio** pour le développement, l'entraînement et le déploiement.
- Modèle capable de traiter des données volumineuses et variées.

## Fonctionnalités ✨

- **Préparation des Données** : Nettoyage, transformation et normalisation des données de carte de crédit. (Voir le tableau joint, c'est déjà fait!)
- **Entraînement du Modèle** : Utilisation de XGBoost pour entraîner un modèle de classification.
- **Optimisation des Hyperparamètres** : Recherche des meilleurs paramètres pour améliorer les performances du modèle.
- **Déploiement** : Hébergement du modèle sur un endpoint SageMaker pour l'inférence en temps réel.
- **Évaluation** : Analyse des performances du modèle sur des jeux de données de test.

## Structure du Projet 🏗️

### Architecture
Le projet est structuré en plusieurs étapes :
1. **Préparation des Données** : Nettoyage, transformation et normalisation des données de carte de crédit.
2. **Entraînement du Modèle** : Utilisation de XGBoost pour entraîner un modèle de classification.
3. **Optimisation des Hyperparamètres** : Recherche des meilleurs paramètres pour améliorer les performances du modèle.
4. **Déploiement** : Hébergement du modèle sur un endpoint SageMaker pour l'inférence en temps réel.
5. **Évaluation** : Analyse des performances du modèle sur des jeux de données de test.

## Réalisation 🛠️

### Préparation des Données
- **Nettoyage des données** pour éliminer les valeurs manquantes et les anomalies.
- **Transformation des variables** pour standardiser les entrées.
- **Normalisation** pour assurer une échelle uniforme des caractéristiques.

### Entraînement et Optimisation
- **Séparation des données** en ensembles d'entraînement et de test.
- **Entraînement initial** du modèle sur l'ensemble d'entraînement.
- **Optimisation des hyperparamètres** à l'aide de techniques comme la recherche en grille.

### Déploiement
- **Création d'un endpoint** SageMaker pour héberger le modèle.
- **Configuration de l'inférence en temps réel** pour prédire les défauts de paiement.

### Évaluation
- **Précision** : 81.5%
- **Rappel** : 68%
- **Graphiques des performances** : Présentation des métriques de performance sous forme graphique.
- **Importance des caractéristiques** : Analyse des variables les plus influentes.

## Structure du Code 🧩
Dans le fichier Détection_localisation_tumeurs.py, comme suit:
│ ├── data_cleaning.py
│ ├── data_transformation.py
│ └── data_normalization.py
├── model_training
│ ├── train_model.py
│ └── optimize_hyperparameters.py
├── deployment
│ ├── deploy_model.py
│ └── inference.py
├── evaluation
│ ├── evaluate_model.py
│ └── performance_metrics.py


## Difficultés Rencontrées 🚧

- **Pré-traitement des données** : Gestion des valeurs manquantes et des anomalies dans les données.
- **Optimisation des hyperparamètres** : Trouver les meilleurs paramètres pour maximiser la performance du modèle.
- **Déploiement** : Configuration et gestion du endpoint SageMaker pour l'inférence en temps réel.

## Améliorations Futures 🔮

- **Amélioration du modèle** : Intégrer des données supplémentaires et tester d'autres algorithmes.
- **Déploiement à grande échelle** : Intégration dans des systèmes de gestion de risques bancaires.
- **Surveillance et mise à jour** : Mettre en place des mécanismes pour surveiller et mettre à jour régulièrement le modèle.
- **Automatisation** : Automatiser le pipeline de déploiement pour réduire les interventions manuelles.

## Frameworks et Technologies Utilisés 🛠️

- **AWS SageMaker Studio** : Plateforme pour le développement, l'entraînement et le déploiement de modèles de machine learning.
- **XGBoost** : Algorithme de gradient boosting performant pour les tâches de classification.
- **Pandas** : Bibliothèque pour la manipulation et l'analyse des données.
- **Scikit-learn** : Bibliothèque pour les tâches de machine learning et d'évaluation de modèles.
- **NumPy** : Bibliothèque pour les opérations mathématiques et les manipulations de tableaux.

## Auteurs et Contributions ✨

Nous remercions tous ceux qui ont contribué à ce projet sur GitHub et qui m'ont aidé à résoudre plusieurs problèmes que j'ai fait face.

## Références et Ressources 📚

- [Documentation Amazon SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

![](https://i.imgur.com/waxVImv.png)


#MachineLearning #CreditRisk #DataScience #AWS #SageMaker #XGBoost #BigData #AI #ArtificialIntelligence #DataPreprocessing #ModelDeployment #HyperparameterTuning #FinancialRisk #Banking #PredictiveModeling #Python #DataAnalytics #ML #FinTech #CreditCard #RiskAssessment #DataCleaning #FeatureEngineering #ModelEvaluation #RealTimeInference

