#!/usr/bin/env python3
# coding=utf-8
import os
import subprocess
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

    
def get_features(info_list, config):
    open('awkward', 'w+').close()
    base_command = config + ' -I {0} -O {1} -nologfile'
    command = []
    command_len = 0
    for p in info_list:
        temp_command = base_command.format(p.file_path, 'awkward')
        if os.name == 'nt':
            command_len += len(temp_command)
            if command_len > 8000:
                final_command = COM_SEP.join(command)
                subprocess.check_output(final_command, shell=True, stderr=subprocess.STDOUT)
                command = []
                command_len = len(temp_command)
        command.append(temp_command)
    final_command = COM_SEP.join(command)
    subprocess.check_output(final_command, shell=True, stderr=subprocess.STDOUT)
    features = []
    with open('awkward') as f:
        for s in f:
            features.append(list(map(float, s.split(',')[1: -1])))
    open('awkward', 'w+').close()
    return features

if __name__ == '__main__':
    print(get_features(datasets.get_ravdess(), EMO2010))
