#!/usr/bin/env python3
# coding=utf-8
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import datasets


if os.name == 'nt':
    COM_SEP = ' & '
    BIN_PATH = 'Win32'
else:
    COM_SEP = ' ; '
    BIN_PATH = 'linux_x64_standalone_static'
smileextract_path = os.path.join('opensmile-2.3.0', 'bin', BIN_PATH, 'SMILExtract_Release.exe')
config_path = 'myconfig'
EMOBASE = '{0} -C {1}'.format(smileextract_path, os.path.join(config_path, 'emobase.conf'))
EMOLARGE = '{0} -C {1}'.format(smileextract_path, os.path.join(config_path, 'emo_large.conf'))
EMO2010 = '{0} -C {1}'.format(smileextract_path, os.path.join(config_path, 'emobase2010.conf'))

AGGREGATES_PATH = 'data_csv/aggregates.csv'
AGGREGATES_COPY_PATH = 'data_csv/aggregates_copy.csv'


def get_output_path(input_path: str) -> str:
    parts = list(Path(input_path).parts)
    parts[parts.index('data')] = 'data_csv'
    return os.path.join(*parts) + '.csv'


def get_features(info_list: Iterable[datasets.FileInfo], config_path: str) -> None:
    base_command = config_path + ' -I "{input}" -O "{output}" -A "{aggregates}" -N {line_name} -nologfile'
    for info in info_list:
        command = base_command.format(
            aggregates=AGGREGATES_PATH, input=info.file_path,
            output=get_output_path(info.file_path), line_name=info.file_path
        )
        subprocess.check_output(command, shell=True)


def mirror_dir_tree(base_path: str) -> None:
    def ignore(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]

    shutil.copytree(base_path, f'{base_path}_csv', ignore=ignore)


def process_aggreagtes() -> None:
    sets = {
        f.file_path: f
        for ds in (datasets.get_emovo(), datasets.get_ravdess(), datasets.get_cafe(), datasets.get_berlin(), datasets.get_tess())
        for f in ds
    }

    with open(AGGREGATES_PATH) as f, open(AGGREGATES_COPY_PATH, 'w+') as new:
        it = iter(f)
        next(f)

        for line in it:
            index = line.index(';')
            path = line[:index].strip("'")
            if not path.endswith('wav'):
                continue

            info = sets[path]
            new_line = (
                # sry for this
                "'" + path[len(r'C:\Projects\Emotions\learning\\') - 1:] +
                f"';{str(info.emotion.value)};{info.dataset};{info.actor};{info.gender.value};{info.intensity.value};{info.age.value}" +
                line[index:]
            )

            new.write(new_line)


if __name__ == '__main__':
    try:
        mirror_dir_tree('data')
    except Exception:
        pass

    print(get_features(datasets.get_emovo(), EMO2010))
    print(get_features(datasets.get_ravdess(), EMO2010))
    print(get_features(datasets.get_cafe(), EMO2010))
    print(get_features(datasets.get_berlin(), EMO2010))
    print(get_features(datasets.get_tess(), EMO2010))

    process_aggreagtes()


