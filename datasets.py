#!/usr/bin/env python3
# coding=utf-8
import os
from enum import Enum
from itertools import zip_longest
from typing import NamedTuple, List


class Emotion(Enum):
    DISGUST = 0
    JOY = 1
    FEAR = 2
    ANGER = 3
    SURPRISE = 4
    SADNESS = 6
    NEUTRAL = 7
    BOREDOM = 8
    CALM = 9


class Gender(Enum):
    MALE = 0
    FEMALE = 1


class Intensity(Enum):
    UNKNOWN = 0
    LOW = 1
    HIGH = 2


class ActorAge(Enum):
    UNKNOWN = 0
    YOUNG = 1
    OLD = 2


class FileInfo(NamedTuple):
    file_path: str
    actor: str
    sentence: str
    emotion: Emotion
    gender: Gender
    intensity: Intensity = Intensity.UNKNOWN
    age: ActorAge = ActorAge.UNKNOWN


BASE_DIR = '../data/'
EMOVO = 'EMOVO'
CAFE = 'CaFe'
BERLIN = 'Berlin/wav'
RAVDESS = 'RAVDESS'
TESS = 'TESS'


def strip_extension(file_name: str) -> str:
    index = file_name.rfind('.')
    return file_name[:index]


def get_emovo() -> List[FileInfo]:
    emotions = {
        'dis': Emotion.DISGUST,
        'gio': Emotion.JOY,
        'pau': Emotion.FEAR,
        'rab': Emotion.ANGER,
        'sor': Emotion.SURPRISE,
        'tri': Emotion.SADNESS,
        'neu': Emotion.NEUTRAL,
    }

    files = []
    for gender, dir_pref in ((Gender.MALE, 'm'), (Gender.FEMALE, 'f')):
        for i in range(1, 4):
            for file in os.listdir(os.path.join(BASE_DIR, EMOVO, f'{dir_pref}{i}')):
                emo, actor, sentence = strip_extension(file).split('-')
                files.append(FileInfo(
                    os.path.join(BASE_DIR, EMOVO, file),
                    actor,
                    sentence,
                    emotions[emo],
                    gender
                ))

    return files


def get_cafe() -> List[FileInfo]:
    emotions = {
        'C': Emotion.ANGER,
        'D': Emotion.DISGUST,
        'J': Emotion.JOY,
        'N': Emotion.NEUTRAL,
        'P': Emotion.FEAR,
        'S': Emotion.SURPRISE,
        'T': Emotion.SADNESS,
    }

    intensities = {
        '1': Intensity.LOW,
        '2': Intensity.HIGH
    }

    files = []

    for dir in os.listdir(os.path.join(BASE_DIR, CAFE)):
        dir = os.path.join(BASE_DIR, CAFE, dir)
        if not os.path.isdir(dir):
            continue

        dir_content: List[str] = os.listdir(dir)
        if all(os.path.isdir(os.path.join(dir, subdir)) for subdir in dir_content):
            dir_content = [
                (file, subdir)
                for subdir in dir_content
                for file in os.listdir(os.path.join(dir, subdir))
            ]
        else:
            dir_content = list(zip_longest(dir_content, (), fillvalue=''))

        for file, subdir in dir_content:
            actor, emo, intensity, sentence = strip_extension(file).split('-')
            emo = emotions[emo]
            intensity = intensities.get(intensity, Intensity.UNKNOWN)
            gender = Gender.MALE if int(actor) % 2 == 1 else Gender.FEMALE
            files.append(FileInfo(
                os.path.join(dir, subdir, file),
                actor,
                sentence,
                emo,
                gender,
                intensity
            ))

    return files


def get_berlin() -> List[FileInfo]:
    emotions = {
        'W': Emotion.ANGER,
        'L': Emotion.BOREDOM,
        'E': Emotion.DISGUST,
        'A': Emotion.FEAR,
        'F': Emotion.JOY,
        'T': Emotion.SADNESS,
        'N': Emotion.NEUTRAL,
    }

    genders = {
        '03': Gender.MALE,
        '08': Gender.FEMALE,
        '09': Gender.FEMALE,
        '10': Gender.MALE,
        '11': Gender.MALE,
        '12': Gender.MALE,
        '13': Gender.FEMALE,
        '14': Gender.FEMALE,
        '15': Gender.MALE,
        '16': Gender.FEMALE,
    }

    files = []
    for file in os.listdir(os.path.join(BASE_DIR, BERLIN)):
        actor = file[:2]
        sentence = file[2:5]
        emo = emotions[file[5]]
        gender = genders[actor]
        files.append(FileInfo(
            os.path.join(BASE_DIR, BERLIN, file),
            actor,
            sentence,
            emo,
            gender
        ))

    return files


def get_ravdess() -> List[FileInfo]:
    emotions = {
        '01': Emotion.NEUTRAL,
        '02': Emotion.CALM,
        '03': Emotion.JOY,
        '04': Emotion.SADNESS,
        '05': Emotion.ANGER,
        '06': Emotion.FEAR,
        '07': Emotion.DISGUST,
        '08': Emotion.SURPRISE,
    }
    intensities = {
        '01': Intensity.LOW,
        '02': Intensity.HIGH,
    }

    files = []
    dir = os.path.join(BASE_DIR, RAVDESS)
    for subdir in os.listdir(dir):
        for file in os.listdir(os.path.join(dir, subdir)):
            _, _, emo, intensity, sentence, _, actor = strip_extension(file).split('-')
            gender = Gender.MALE if int(actor) % 2 == 1 else Gender.FEMALE
            files.append(FileInfo(
                os.path.join(dir, subdir, file),
                actor,
                sentence,
                emotions[emo],
                gender,
                intensities[intensity]
            ))

    return files


def get_tess() -> List[FileInfo]:
    emotions = {
        'Angry': Emotion.ANGER,
        'Disgust': Emotion.DISGUST,
        'Fear': Emotion.FEAR,
        'Happy': Emotion.JOY,
        'Neutral': Emotion.NEUTRAL,
        'Sad': Emotion.SADNESS,
        'Pleasant Surprise': Emotion.SURPRISE,
    }
    ages = {
        'Younger': ActorAge.YOUNG,
        'Older': ActorAge.OLD,
    }

    files = []
    dir = os.path.join(BASE_DIR, TESS)
    for subdir in os.listdir(dir):
        if not os.path.isdir(os.path.join(dir, subdir)):
            continue

        # :(
        if '_' not in subdir:
            age = 'Older'
            emo = 'Pleasant Surprise'
        else:
            age, emo = subdir.split(' ')[0], subdir.split('_')[1]
        age = ages[age]
        emo = emotions[emo]
        for file in os.listdir(os.path.join(dir, subdir)):
            _, sentence, _ = file.split('_')
            files.append(FileInfo(
                os.path.join(dir, subdir, file),
                age.name,
                sentence,
                emo,
                Gender.MALE,
                age=age
            ))

    return files
