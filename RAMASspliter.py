import os
import csv
import operator
import pydub


ANNOTATION_PATH = 'Annotations_by_files'
DATA_PATH = os.path.join('Data', 'Audio')
ANNOTATIONRES_PATH = 'Annotation_summary'
DATARES_PATH = os.path.join('Data', 'AudioSplitted')


def main():
    for file in os.listdir(ANNOTATION_PATH):
        header, dct = read(file)
        normalize(dct)
        write_annres(file, header, dct)
        export_split(file, dct)


def read(file):
    with open(os.path.join(ANNOTATION_PATH, file)) as f:
        reader = csv.reader(f)
        header = next(reader)
        dct = {}
        for row in reader:
            temp = list(map(float, row))
            if temp[0] in dct:
                dct[temp[0]] = list(map(operator.add, dct[temp[0]], temp[1:-1]))
            else:
                dct[temp[0]] = temp[1:-1]
    return header, dct


def normalize(dct):
    to_del = []
    for k, val in dct.items():
        s = sum(val)
        if s != 0:
            val = list(map(lambda x: x / s, val))
            dct[k] = val
        else:
            to_del.append(k)
    for el in to_del:
        del dct[el]


def write_annres(file, header, dct):
    with open(os.path.join(ANNOTATIONRES_PATH, file), 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header[:-1])
        writer.writerows(
            map(operator.add, list(map(lambda x: [x], dct.keys())), list(dct.values()))
        )


def export_split(file, dct):
    heh = file.split('_')
    heh.pop()
    heh.append('mic.wav')
    wav = pydub.AudioSegment.from_wav(os.path.join(DATA_PATH, '_'.join(heh)))
    first = list(dct.keys())
    second = (list(dct.keys())[1:])
    second.append(second[-1] + 1)
    segments = zip(first, second)
    heh[-1] = None
    heh.append('mic.wav')
    for seg in segments:
        heh[-2] = str(seg[0])
        if seg[1] - seg[0] > 1:
            seg = (seg[0], seg[0] + 1)
        to_export = wav[seg[0] * 1000: seg[1] * 1000]
        if seg[1] - seg[0] < 1:
            to_export += pydub.AudioSegment.silent((seg[1] - seg[0]) * 1000)
        to_export.export(
            os.path.join(DATARES_PATH, '_'.join(heh)),
            format='wav'
        )


if __name__ == '__main__':
    main()
