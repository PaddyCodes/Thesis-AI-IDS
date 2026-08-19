# Random Forest AI-IDS Model Record

## Final Model

Classifier: RandomForestClassifier

Predictors: 69

Hyperparameters:

- n_estimators: 100
- max_depth: 20
- min_samples_split: 2
- min_samples_leaf: 2
- max_features: 0.5
- class_weight: balanced_subsample
- random_state: 42

## Model Selection

The final configuration was selected using RandomizedSearchCV.

- Search iterations: 10
- Cross-validation: 5-fold stratified
- Optimisation metric: F1-score
- Best mean CV F1: 0.997419

## Validation Performance

- Accuracy: 0.999143
- Precision: 0.995431
- Recall: 0.999515
- F1-score: 0.997468
- False Positive Rate: 0.000932
- False Negative Rate: 0.000485

Confusion matrix:

- TN: 313,966
- FP: 293
- FN: 31
- TP: 63,830

## Integrity

random_forest.joblib

SHA256:
EC2DBF6482DBFB4821D521528673F5CC4357FE9E973D5783B0C673A5DE23374C

random_forest_features.txt

SHA256:
E70C12C8D468B7B1EC76DF559EF823AF3B89B4CF018B0C9C1CF38DE0291BDAAF

## Experimental Status

The model was frozen before final held-out test evaluation and before
construction of the AI-modified malicious test condition.

The held-out test partition has not been used for model training,
hyperparameter selection or validation.