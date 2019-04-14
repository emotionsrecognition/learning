import os
import json
import time
import pickle

from collections import defaultdict
from functools import partial
from typing import Union

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score


Data = Union[np.ndarray, pd.DataFrame]
Target = Union[np.ndarray, pd.Series]

EXPERIMENT_FOLDER = 'experiment'
FILE_DIR = os.path.split(__file__)[0]
METRICS = {
    'accuracy': accuracy_score,
    'precision_macro': partial(precision_score, average='macro'),
    'recall_macro': partial(recall_score, average='macro'),
    'f1_macro': partial(f1_score, average='macro'),
    'r2': r2_score
}


# todo: clf_params instead clf
# todo: test Model classes


class Model:
    def __init__(self, model, model_class):
        self.model = model
        self.model_class = model_class

    def fit(self, X: Data, y: Target):
        self.model.fit(X, y)

    def predict(self, X: Data) -> np.ndarray:
        return self.model.predict(X)

    def reset(self) -> 'Model':
        self.model = self.get_clf_class()(**self.model.get_params())
        return self

    def get_clf_class(self):
        return self.model_class

    def save_model(self, path: str):
        raise NotImplemented

    def load_model(self, path: str) -> 'Model':
        raise NotImplemented


class ClassifierPickleSaveLoad(Model):
    def __init__(self, model, model_class):
        super().__init__(model, model_class)

    def save_model(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, path: str) -> 'Model':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return ClassifierPickleSaveLoad(model, self.model_class)


class ClassifierWithSaveLoadInterface(Model):
    def __init__(self, model, model_class):
        super().__init__(model, model_class)

    def save_model(self, path: str):
        self.model.save_model(path)

    def load_model(self, path: str):
        model = self.get_clf_class()
        model.load_model(path)
        return model


def run_model_pipeline_cv(X: Data, y: Target, clf_model: Model,
                          experiment_name: str, n_folds: int = 5):
    # It needs because of iloc    
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Training preparation
    exp_path = os.path.join(FILE_DIR, EXPERIMENT_FOLDER, experiment_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    predictions = np.zeros(y.shape) - 1
    scores, models, durations = defaultdict(list), [], []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        start = time.monotonic()

        # split data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # train clf
        clf = clf_model.reset()
        clf.fit(X_train, y_train)
        models.append(clf)
        durations.append(time.monotonic() - start)

        # result evaluation
        y_pred = clf.predict(X_test)
        predictions[test_index] = y_pred
        for metric_name, metric_scorer in METRICS.items():
            scores[metric_name].append(metric_scorer(y_test, y_pred))

        # saving intermediate results
        clf.save_model(os.path.join(exp_path, 'model_{}'.format(i)))

        scores_path = os.path.join(exp_path, 'scores.json')
        with open(scores_path, 'w') as f:
            json.dump(scores, f)

        predictions_path = os.path.join(exp_path, 'predictions.pkl')
        with open(predictions_path, 'bw') as f:
            pickle.dump(predictions, f)

        durations_path = os.path.join(exp_path, 'durations.pkl')
        with open(durations_path, 'bw') as f:
            pickle.dump(durations, f)

    return predictions, scores, models, durations


def example():
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data['data'], data['target']

    model = ClassifierPickleSaveLoad(LGBMClassifier(), LGBMClassifier)
    pipeline_results = run_model_pipeline_cv(X, y, model, 'example_exp')
    predictions, scores, models, durations = pipeline_results


if __name__ == '__main__':
    example()
