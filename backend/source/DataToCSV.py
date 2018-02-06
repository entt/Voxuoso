import csv
from os import listdir
from os.path import (
    isfile,
    join,
    abspath,
    dirname,
)

from FeatureExtraction import FeatureExtraction

audio_dir = join(abspath(dirname(__file__)), 'data/Voices/')
csv_dir = join(abspath(dirname(__file__)), 'data/Inputs.csv')

audio_files = [join(audio_dir, f)
               for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

fields = [
    'file_name',
    'ZCR', 'EnergyEntropy',
    'SpectralCentroid', 'SpectralSpread',
    'SpectralEntropy', 'SpectralFlux'
]

with open(csv_dir, 'w+') as write:
    writer = csv.DictWriter(write, fieldnames=fields)
    writer.writeheader()
    FE = FeatureExtraction()

    for line in write:
        for file in audio_files:
            print(line, file)
            print 'Current file: ' + file.split('/')[-1]
            features = FE.feature_extraction(file)
            writer.writerow({
                'file_name': file.split('/')[-1],
                'ZCR': features[0],
                'EnergyEntropy': features[2],
                'SpectralCentroid': features[3],
                'SpectralSpread': features[4],
                'SpectralEntropy': features[5],
                'SpectralFlux': features[6]
            })
