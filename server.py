#!/usr/bin/env python3
# coding=utf-8
import asyncio as aio
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional, NamedTuple, Tuple, Iterable

from sanic import Sanic
from sanic.request import Request
from sanic.response import json, HTTPResponse

import pandas as pd
from sklearn.externals import joblib

from features import get_features, CONFIG
from datasets import Emotion, Gender


PARALLELISM_LEVEL = 4
GENDER_MODEL_PATH = 'gender.model'
MALE_MODEL_PATH = 'male.model'
FEMALE_MODEL_PATH = 'female.model'
app = Sanic()


class PredictionResult(NamedTuple):
    gender: Gender
    emotion: Emotion

    def to_tuple(self) -> Tuple[str, str]:
        return self.gender.name, self.emotion.name


@app.listener('before_server_start')
def init(_, loop: aio.AbstractEventLoop):
    global semaphore
    semaphore = aio.Semaphore(PARALLELISM_LEVEL, loop=loop)


@app.post('/process_file')
async def process_file_async(request: Request):
    file_path = request.body.decode('utf8')
    if not file_path:
        return HTTPResponse(status=400)

    with provide_throttling() as throttling:
        await throttling
        features_path = await extract_features_async(file_path)
        if not features_path:
            return HTTPResponse(status=500)

        results = await run_model_async(features_path)

        return json({
            'features_path': features_path,
            'data': list(r.to_tuple() for r in results)
        })


@contextmanager
def provide_throttling() -> aio.Future:
    try:
        yield semaphore.acquire()
    finally:
        semaphore.release()


async def extract_features_async(file_path: str) -> Optional[str]:
    return await aio.get_event_loop().run_in_executor(executor, partial(extract_features_internal, file_path))


def extract_features_internal(file_path: str) -> Optional[str]:
    features_path = file_path.rsplit('.', maxsplit=1)[0] + '.csv'
    if get_features([file_path], CONFIG, features_path):
        return features_path

    return None


async def run_model_async(features_path: str) -> Iterable[PredictionResult]:
    return await aio.get_event_loop().run_in_executor(executor, partial(run_model_internal, features_path))


def run_model_internal(features_path: str) -> Iterable[PredictionResult]:
    df = pd.read_csv(features_path, sep=';', encoding='latin1').iloc[:, 2:]
    genders = gender_model.predict(df)
    df.insert(0, column='gender', value=genders)

    male_df = df.loc[df['gender'] == Gender.MALE.value].iloc[:, 1:]
    female_df = df.loc[df['gender'] == Gender.FEMALE.value].iloc[:, 1:]

    male_rows = iter(male_model.predict(male_df))
    female_rows = iter(female_model.predict(female_df))

    result = []

    for gender in genders:
        if gender == Gender.MALE.value:
            result.append(PredictionResult(Gender.MALE, Emotion(next(male_rows))))
        else:
            result.append(PredictionResult(Gender.FEMALE, Emotion(next(female_rows))))

    return result


if __name__ == '__main__':
    executor = ThreadPoolExecutor()
    gender_model = joblib.load(GENDER_MODEL_PATH)
    male_model = joblib.load(MALE_MODEL_PATH)
    female_model = joblib.load(FEMALE_MODEL_PATH)

    app.run(host='0.0.0.0', port=8000)
