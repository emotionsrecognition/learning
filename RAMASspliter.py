import os
import csv
import operator
import random
from collections import OrderedDict
from pprint import pprint
from typing import Tuple, Dict, Sequence, NamedTuple

import pydub
from datetime import timedelta

from datasets import Emotion

ANNOTATION_PATH = os.path.join('RAMAS', 'Annotations_by_files')
DATA_PATH = os.path.join('RAMAS', 'Data', 'Audio')
ANNOTATIONRES_PATH = os.path.join('RAMAS', 'Annotation_summary')
DATARES_PATH = os.path.join('RAMAS', 'Data', 'AudioSplitted')


TimeCodedData = Dict[float, Sequence[float]]
Header = Sequence[str]
FileData = Tuple[Header, TimeCodedData]


class EmotionSegment(NamedTuple):
    start_time: float
    end_time: float
    emotion: Emotion


def get_emotion(description: str) -> Emotion:
    return {
        'Angry': Emotion.ANGER,
        'Sad': Emotion.SADNESS,
        'Disgusted': Emotion.DISGUST,
        'Happy': Emotion.JOY,
        'Scared': Emotion.FEAR,
        'Surprised': Emotion.SURPRISE,
        'Neutral': Emotion.NEUTRAL
    }.get(description, Emotion.UNKNOWN)


def main() -> None:
    for file in os.listdir(ANNOTATION_PATH):
        header, times_dict = read(file)
        segments = vote(header, times_dict)
        # write_annres(file, header, times_dict)
        export_split(file, segments)


def read(file) -> FileData:
    with open(os.path.join(ANNOTATION_PATH, file)) as f:
        reader = csv.reader(f)
        header = next(reader)
        times_dict = OrderedDict()
        for row in reader:
            timecode, *values, _ = map(float, row)
            stored_values = times_dict.get(timecode)
            if stored_values is None:
                times_dict[timecode] = values
            else:
                times_dict[timecode] = [a + b for a, b in zip(stored_values, values)]

    return header, times_dict


def vote(header: Header, times_dict: TimeCodedData) -> Sequence[EmotionSegment]:
    processed_data = []
    start_timecode = None
    last_emotion_index = None

    def append_segment(start_timecode, end_timecode, emotion_index):
        if emotion_index is None:
            return
        emotion = get_emotion(header[emotion_index + 1])
        if emotion != Emotion.UNKNOWN:
            processed_data.append(EmotionSegment(start_timecode, end_timecode, emotion))

    for timecode, row in times_dict.items():
        row = row[:7]  # cut off unnecessary emotions
        row_max = max(row)
        if not row_max:
            emotion_indices = set()
        else:
            emotion_indices = set(i for i, x in enumerate(row) if x == row_max)

        if last_emotion_index not in emotion_indices:
            if start_timecode is not None:
                append_segment(start_timecode, timecode, last_emotion_index)

            last_emotion_index = (emotion_indices or None) and random.choice(tuple(emotion_indices))
            start_timecode = timecode

    append_segment(start_timecode, timecode, last_emotion_index)

    return processed_data


def write_annres(file, header, dct):
    with open(os.path.join(ANNOTATIONRES_PATH, file), 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header[:-1])
        writer.writerows(
            map(operator.add, list(map(lambda x: [x], dct.keys())), list(dct.values()))
        )


def export_split(file: str, segments: Sequence[EmotionSegment]):
    # pprint([
    #     (str(timedelta(seconds=s)), str(timedelta(seconds=e)), s, e, em)
    #     for s, e, em in segments
    # ])
    fname_base, _ = file.rsplit('_', maxsplit=1)
    source_wav_name = '_'.join((fname_base, 'mic.wav'))

    wav = pydub.AudioSegment.from_wav(os.path.join(DATA_PATH, source_wav_name))

    for i, segment in enumerate(segments):
        wav[segment.start_time * 1000:segment.end_time * 1000 + 1].export(
            os.path.join(DATARES_PATH, '_'.join((fname_base, str(i), str(segment.emotion.value), 'mic.wav'))),
            format='wav'
        )


if __name__ == '__main__':
    main()
