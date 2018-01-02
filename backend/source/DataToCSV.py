import csv
from os import listdir
from os.path import (
    isfile,
    join,
    abspath,
    dirname,
)

import FeatureExtraction as FE

module_directory = abspath(dirname(__file__))
audio_dir = join(module_directory, './data/Voices/')
csv_dir = join(module_directory, './data/Inputs.csv')

audio_files = [join(audio_dir, f)
               for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

fields = [
    'file_name',
    'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4',
    'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8',
    'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
    'MFCC13'
]

with open(csv_dir, 'w+') as write:
    writer = csv.DictWriter(write, fieldnames=fields)
    writer.writeheader()

    for line in write:
        for file in audio_files:
            print 'Current file: ' + file.split('/')[-1]
            features = FE.feature_extraction(file)
            writer.writerow({
                'file_name': file.split('/')[-1],
                'MFCC1': features[8],
                'MFCC2': features[9],
                'MFCC3': features[10],
                'MFCC4': features[11],
                'MFCC5': features[12],
                'MFCC6': features[13],
                'MFCC7': features[14],
                'MFCC8': features[15],
                'MFCC9': features[16],
                'MFCC10': features[17],
                'MFCC11': features[18],
                'MFCC12': features[19],
                'MFCC13': features[20]
            })
