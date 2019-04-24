import os
import json
import time
import pickle

from collections import defaultdict
from functools import partial
from typing import Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.externals import joblib


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

    def get_features_importance(self):
        # shit code
        if self.get_clf_class() == LGBMClassifier:
            return self.model.feature_importances_
        if self.get_clf_class() == CatBoostClassifier:
            return self.model.get_feature_importance()
        if self.get_clf_class() == XGBClassifier:
            return self.model.feature_importance_

        raise NotImplemented

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


class DatasetsFolds:
    def split(self, X, y):
        """

        :param X: data for model training
        :param y: column with dataset names
        :return: Iterable([train_indices, test_indices)
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError('X.shape[0] != y.shape[0]')

        y = pd.Series(y)
        for y_unique_value in y.unique():
            test_part = y[y == y_unique_value]

            train_indices = y.index.difference(test_part.index)
            test_indices = test_part.index
            yield train_indices, test_indices


def save_model_and_artifacts(model, i, scores, predictions, durations, exp_path):
    model.save_model(os.path.join(exp_path, 'model_{}'.format(i)))

    scores_path = os.path.join(exp_path, 'scores.json')
    with open(scores_path, 'w') as f:
        json.dump(scores, f)

    predictions_path = os.path.join(exp_path, 'predictions.pkl')
    with open(predictions_path, 'bw') as f:
        pickle.dump(predictions, f)

    durations_path = os.path.join(exp_path, 'durations.pkl')
    with open(durations_path, 'bw') as f:
        pickle.dump(durations, f)


def get_features_importance(models):
    importances = [model.get_features_importance() for model in models]
    importances = np.array(importances)

    sum_importance = importances.sum(axis=0)
    return sum_importance, sum_importance / sum_importance.max()


def plt_bar_feature_importance_with_bad_indication(
        feature_names, feature_freqs, n, exp_path):
    y_pos = np.arange(len(feature_names))

    bad_poses, bad_names, bad_freqs = y_pos[:n], feature_names[:n], feature_freqs[:n]
    good_poses, good_names, good_freqs = y_pos[n:], feature_names[n:], feature_freqs[n:]

    plt.bar(bad_poses, bad_freqs, color='r', align='center')
    plt.xticks(bad_poses, bad_names, rotation='vertical')

    plt.bar(good_poses, good_freqs, color='g', align='center')
    plt.xticks(good_poses, good_names, rotation='vertical')

    plt.ylabel('Frequency by maximum')
    plt.title('Feature name')

    plt.savefig(os.path.join(exp_path, 'feature_importance_with_bad_indication.jpg'))


def save_plt_bar_feature_importance(feature_names, feature_freqs, exp_path):
    plt.figure(figsize=(10, 20))
    y_pos = np.arange(len(feature_names))

    plt.bar(y_pos, feature_freqs, color='b', align='center')
    plt.xticks(y_pos, feature_names, rotation='vertical')

    plt.ylabel('Frequency (sum / max)')
    plt.title('Feature name')

    plt.savefig(os.path.join(exp_path, 'feature_importance.jpg'))


def get_top_worst_features(models, feature_names, n, exp_path):
    _, max_freq_importance = get_features_importance(models)
    name_max_freq_triples = [(i, feature_names[i], max_freq)
                             for i, max_freq in enumerate(max_freq_importance)]
    name_max_freq_triples.sort(key=lambda x: x[2])

    freqs = [freq for _, _, freq in name_max_freq_triples]
    names = [name for _, name, _ in name_max_freq_triples]

    plt_bar_feature_importance_with_bad_indication(names, freqs, n, exp_path)

    return name_max_freq_triples[:n]


def split_data(X, y, train_index, test_index):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    return X_train, X_test, y_train, y_test


def run_model_pipeline_cv(X: Data, y: Target, feature_names: List[str],
                          clf_model: Model, experiment_name: str,
                          n_folds: int = 5, cv_datasets_column=None,
                          ):
    # It needs because of iloc
    if feature_names:
        X = pd.DataFrame(X, columns=feature_names)
    else:
        X = pd.DataFrame(X)
    y = pd.Series(y)

    # Training preparation
    exp_path = os.path.join(FILE_DIR, EXPERIMENT_FOLDER, experiment_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    predictions = pd.Series(np.zeros(y.shape) - 1, index=y.index)
    scores, models, durations = defaultdict(list), [], []

    if cv_datasets_column is not None:
        stratified_column = cv_datasets_column
        data_splitter = DatasetsFolds()

    else:
        stratified_column = y
        data_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for i, (train_index, test_index) in enumerate(data_splitter.split(X, stratified_column)):
        start = time.monotonic()

        X_train, X_test, y_train, y_test = split_data(X, y, train_index, test_index)

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
        save_model_and_artifacts(clf, i, scores, predictions, durations, exp_path)

    # calc mean metrics
    for metric_name, metric_scorer in METRICS.items():
        scores[metric_name + '_mean'] = np.array(scores[metric_name]).mean()

    # calc features importance
    _, sum_divide_max_importance = get_features_importance(models)
    id_name_freq_triples = [(i, feature_names[i], freq)
                            for i, freq in enumerate(sum_divide_max_importance)]
    id_name_freq_triples.sort(key=lambda x: x[2])  # x[2] == freq

    # save features importance
    id_name_freq_triples_path = os.path.join(exp_path, 'features_id_name_freq_triples.joblib')
    joblib.dump(id_name_freq_triples, id_name_freq_triples_path)

    _, features_names, features_freqs = zip(*id_name_freq_triples)  # transpose id_name_freq_triples
    save_plt_bar_feature_importance(features_names, features_freqs, exp_path)

    return predictions, scores, models, durations, id_name_freq_triples


def example():
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data['data'], data['target']

    model = ClassifierPickleSaveLoad(LGBMClassifier(), LGBMClassifier)
    pipeline_results = run_model_pipeline_cv(X, y, data['feature_names'],
                                             model, 'example_exp')

    predictions, scores, models, durations, id_name_freq_triples = pipeline_results


def example_with_cv_by_datasets():
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data['data'], data['target']
    cv_column = pd.Series(y).index.values % 4

    model = ClassifierPickleSaveLoad(LGBMClassifier(), LGBMClassifier)
    pipeline_results = run_model_pipeline_cv(X, y, data['feature_names'],
                                             model, 'example_exp_cv_by_datasets',

                                             # use column with datasets names instead "data['target']"
                                             cv_datasets_column=cv_column)

    predictions, scores, models, durations, id_name_freq_triples = pipeline_results


def example_with_feature_selection():
    from sklearn.datasets import load_iris
    data = load_iris()
    X, y = data['data'], data['target']

    # load previous features
    previous_features_path = os.path.join(FILE_DIR, EXPERIMENT_FOLDER, 'example_exp',
                                          'features_id_name_freq_triples.joblib')
    features_id_name_freq_triples = joblib.load(previous_features_path)

    # select data without bad features
    features_ids, features_names, features_freqs = zip(*features_id_name_freq_triples)

    # delete first features (small importance)
    good_features_names = list(features_names[2:])

    X = pd.DataFrame(X, columns=data['feature_names'])
    X = X.loc[:, good_features_names]

    # learning
    model = ClassifierPickleSaveLoad(LGBMClassifier(), LGBMClassifier)
    pipeline_results = run_model_pipeline_cv(X, y, good_features_names,
                                             model, 'example_exp_without_some_features')

    predictions, scores, models, durations, id_name_freq_triples = pipeline_results


def real_case_example():
    data = pd.read_csv('shrinked2.csv', sep=';', encoding='latin1')
    data = data.loc[data['gender'] == 0]
    X, y = data.iloc[:, 8:], data['emotion']
    cv_column = data['dataset']
    
    model = ClassifierPickleSaveLoad(LGBMClassifier(), LGBMClassifier)
    pipeline_results = run_model_pipeline_cv(X, y, list(X.columns),
                                             model, 'real_case_example',
                                             cv_datasets_column=cv_column)

    predictions, scores, models, durations, id_name_freq_triples = pipeline_results


if __name__ == '__main__':
    real_case_example()
    # example()
