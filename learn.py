#!/usr/bin/env python3
# coding=utf-8
import math
from itertools import chain
import pandas as pd
import numpy as np
from datasets import FileInfo, Emotion, Gender, ActorAge, Intensity, get_berlin, get_cafe, get_emovo, get_ravdess, \
    get_tess

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def select_features(df):
    return df.loc[:, df.columns != 'emotion']


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


if __name__ == '__main__':
    df = pd.DataFrame()

    for info in filter(filter_function, chain(get_ravdess(), get_berlin(), get_cafe(), get_emovo())):
        features = pd.read_csv(info.features_path, sep=';').iloc[:, 2:]
        # TODO: how to insert int column in this fucking pandas?
        features.insert(0, 'emotion', info.emotion.value, True)

        df = df.append(features)

    data_train, data_test, answer_train, answer_test = train_test_split(select_features(df), df['emotion'], test_size=0.3, shuffle=True, random_state=20)

    forest = RandomForestClassifier(n_jobs=-1)
    forest_params = dict(
        n_estimators=[1000, 5000],
        max_depth=[7, 17, 50],
        max_features=[1, 2, 3, 5, 10, 32, None],
        random_state=[42],
    )
    grid = GridSearchCV(
        forest, forest_params,
        cv=7, n_jobs=-1,
        verbose=True
    )

    grid.fit(data_train, answer_train)

    print("Best parameters set found on development set:")
    print()
    print(grid.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = answer_test, grid.predict(data_test)
    print(classification_report(y_true, y_pred))
    print()

    validate(grid, data_test, answer_test)

    joblib.dump(grid.best_estimator_, 'grid2.pkl')
