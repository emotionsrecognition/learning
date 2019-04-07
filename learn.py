#!/usr/bin/env python3
# coding=utf-8
import math
import pickle
from itertools import chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import FileInfo, Emotion, Gender, ActorAge, Intensity, get_berlin, get_cafe, get_emovo, get_ravdess, \
    get_tess

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def select_features(df):
    return df.drop(['emotion', 'F0finEnv_sma'], axis=1)


def validate(model, data_test, answer_test):
    test_df = pd.concat([data_test, answer_test], axis=1)

    predicted = {}
    accuracies = {}
    for emo in Emotion:
        cat = test_df[test_df['emotion'] == emo.value]
        if not len(cat):
            continue
        predicted[emo] = pred = model.predict(select_features(cat))
        accuracies[emo] = np.mean(pred == cat['emotion'])

    total = 1
    for emo, acc in accuracies.items():
        print(f'{emo}:\t{acc}')
        total *= acc

    overall_accuracy = math.sqrt(total)
    print(overall_accuracy)


def filter_function(info: FileInfo) -> bool:
    return (info.gender == Gender.MALE) and (
                info.intensity == Intensity.HIGH or info.intensity == Intensity.UNKNOWN) and (
                   info.emotion == Emotion.NEUTRAL or info.emotion == Emotion.JOY or info.emotion == Emotion.ANGER or
                   info.emotion == Emotion.SURPRISE or True
           ) and (info.age == ActorAge.OLD or info.age == ActorAge.UNKNOWN)


def features_importance(data: pd.DataFrame, model) -> None:
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    print("Feature ranking:")

    X = select_features(data)
    columns = list(X.columns.values)

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), [columns[i] for i in indices], rotation='vertical')
    plt.xlim([-1, X.shape[1]])
    plt.show()


if __name__ == '__main__':
    df = pd.DataFrame()

    for info in filter(filter_function, chain(get_ravdess(), get_emovo(), get_berlin(), get_cafe())):
        features = pd.read_csv(info.features_path, sep=';').iloc[::7, 2:]
        features.insert(0, 'emotion', info.emotion.value, True)

        df = df.append(features)

    data_train, data_test, answer_train, answer_test = train_test_split(select_features(df), df['emotion'], test_size=0.2, shuffle=True, random_state=20)

    forest = RandomForestClassifier(n_jobs=-1)
    forest_params = dict(
        n_estimators=[1000, 5000],
        max_depth=[3, 7, 17, 30],
        max_features=[3, 5, 10, 20, 50, None],
        random_state=[42],
    )
    grid = GridSearchCV(
        forest, forest_params,
        cv=7, n_jobs=-1,
        verbose=True
    )

    grid.fit(data_train, answer_train)

    y_true, y_pred = answer_test, forest.predict(data_test)
    print(classification_report(y_true, y_pred))

    features_importance(df, grid.best_estimator_)
    # print()
    #
    # with open('forest_special.pkl', 'w+b') as f:
    #     pickle.dump(forest, f, -1)
