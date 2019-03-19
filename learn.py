#!/usr/bin/env python3
# coding=utf-8
import math
import os
from typing import Iterable, NamedTuple, List

conda = 'C:\ProgramData\Anaconda3\envs\emotions_36'
os.environ["PATH"] += rf";{conda}"
os.environ["PATH"] += rf";{conda}\Library\mingw-w64\\bin"
os.environ["PATH"] += rf";{conda}\Library\\usr\\bin"
os.environ["PATH"] += rf";{conda}\Library\\bin"
os.environ["PATH"] += rf";{conda}\Scripts"
import librosa
import pandas as pd
import numpy as np
from datasets import FileInfo, Emotion, ActorAge, Intensity, get_berlin, get_cafe, get_emovo, get_ravdess, get_tess

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV


class FileInfoWithFeatures(NamedTuple):
    info: FileInfo
    mfcc: List


def select_mfcc(df):
    return df.loc[:, df.columns != 0]


def validate(model, data_test, answer_test):
    test_df = pd.concat([data_test, answer_test], axis=1)

    predicted = {}
    accuracies = {}
    for emo in Emotion:
        cat = test_df[test_df[0] == emo.value]
        if not len(cat):
            continue
        predicted[emo] = pred = model.predict(select_mfcc(cat))
        accuracies[emo] = np.mean(pred == cat[0])

    total = 1
    for emo, acc in accuracies.items():
        print(f'{emo}:\t{acc}')
        total *= acc

    overall_accuracy = math.sqrt(total)
    print(overall_accuracy)


def filter_function(info: FileInfo) -> bool:
    return info.age == ActorAge.OLD and (
        True  # and info.emotion == Emotion.NEUTRAL or info.emotion == Emotion.JOY or info.emotion == Emotion.ANGER
    )


def get_features(infos: Iterable[FileInfo]) -> Iterable[FileInfoWithFeatures]:
    for info in infos:
        y, sample_rate = librosa.load(info.file_path)
        mfcc = list(librosa.feature.mfcc(y=y, sr=sample_rate, hop_length=512, n_mfcc=13).reshape(1, -1)[0])
        if not len(mfcc):
            continue
        yield FileInfoWithFeatures(info, mfcc)


def get_dataframe(infos: Iterable[FileInfoWithFeatures]) -> pd.DataFrame:
    data = [
        [info.info.emotion.value] + info.mfcc
        for info in infos
    ]
    return pd.DataFrame(data).fillna(0)


if __name__ == '__main__':
    dataset = filter(filter_function, get_tess())
    features = get_features(dataset)
    df = get_dataframe(features)

    data_train, data_test, answer_train, answer_test = train_test_split(select_mfcc(df), df[0], test_size=0.3, shuffle=True, random_state=16)

    with pd.option_context('display.max_rows', None):
        print(data_test)
    print(answer_test)

    forest = RandomForestClassifier(n_estimators=42, n_jobs=-1, random_state=17)
    forest_params = {
        'max_depth': range(100, 110),
        'max_features': range(900, 910)
    }
    grid = GridSearchCV(
        forest, forest_params,
        cv=7, n_jobs=-1,
        verbose=True
    )

    pipe = Pipeline([
        # ('lr', LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1000))
        ('lr', grid)
    ])

    pipe.fit(data_train, answer_train)
    validate(pipe, data_test, answer_test)

