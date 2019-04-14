#!/usr/bin/env python3
# coding=utf-8
import os
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from typing import Iterable, List, Dict
from operator import attrgetter

import datasets


if os.name == 'nt':
    COM_SEP = ' & '
    BIN_PATH = 'Win32'
else:
    COM_SEP = ' ; '
    BIN_PATH = 'linux_x64_standalone_static'
smileextract_path = os.path.join('opensmile-2.3.0', 'bin', BIN_PATH, 'SMILExtract_Release.exe')
config_path = 'opensmile_config'
CONFIG = f'{smileextract_path} -C {os.path.join("opensmile_config", "main.conf")}'

AGGREGATES_PATH = 'aggr_data.csv'
AGGREGATES_COPY_PATH = 'aggr_data_processed.csv'


def get_output_path(input_path: str) -> str:
    parts = list(Path(input_path).parts)
    parts[parts.index('data')] = 'data_csv'
    return os.path.join(*parts) + '.csv'


def get_features(info_list: Iterable[datasets.FileInfo], config_path: str) -> None:
    processed = open('processed.txt', 'a')
    base_command = config_path + ' -I "{input}" -A "{aggregates}" -N {line_name} -nologfile'
    for info in info_list:
        try:
            command = base_command.format(
                aggregates=AGGREGATES_PATH, input=info.file_path, line_name='_'.join(info.file_path.split(' '))
            )
            subprocess.check_output(command, shell=True)
        except Exception:
            continue
        else:
            processed.write(f'{info.file_path}\n')
    processed.close()


def mirror_dir_tree(base_path: str) -> None:
    def ignore(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    shutil.copytree(base_path, f'{base_path}_csv', ignore=ignore)


def process_aggreagtes(info_list: Iterable[datasets.FileInfo]) -> None:
    sets: Dict[str, datasets.FileInfo] = {
        '_'.join(info.file_path.split(' ')): info
        for info in info_list
    }

    new_columns = (
        ('emotion', attrgetter('emotion.value')),
        ('dataset', attrgetter('dataset')),
        ('actor', attrgetter('actor')),
        ('gender', attrgetter('gender.value')),
        ('intensity', attrgetter('intensity.value')),
        ('age', attrgetter('age.value')),
    )

    with open(AGGREGATES_PATH) as old, open(AGGREGATES_COPY_PATH, 'w+') as new:
        old_iter = iter(old)
        header: List[str] = next(old).split(';')
        new_header = ';'.join(header[:1] + [c for c, _ in new_columns] + header[1:])
        new.write(new_header)

        for line in old_iter:
            second_column_index = line.index(';')
            path = line[:second_column_index].strip("'")
            if not path.endswith('wav'):
                continue

            info = sets[path]
            new_line = (
                f"'{path[len(datasets.BASE_DIR) + 1:]}';" +
                ';'.join(str(getter(info)) for _, getter in new_columns) + line[second_column_index:]
            )

            new.write(new_line)


if __name__ == '__main__':
    try:
        mirror_dir_tree('data')
    except Exception:
        pass

    ds = tuple(chain(
        datasets.get_emovo(), datasets.get_ravdess(), datasets.get_cafe(), datasets.get_berlin(), datasets.get_tess()
    ))

    get_features(ds, CONFIG)
    process_aggreagtes(ds)


