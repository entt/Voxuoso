"""
    Signal processing:
        Feature Extraction:
            - MFCC
    =========================
    (c) Roent Jogno AY 2017 - 2018
"""
from pyAudioAnalysis import (
    audioBasicIO as aIO,
    audioFeatureExtraction as aFE,
)


def feature_extraction(audio):
    """
    Extracts features from audio file.
        -0:         Zero Crossing Rate
        -1:         Energy
        -2:         Entropy of Energy
        -3:         Spectral Centroid
        -4:         Spectral Spread
        -5:         Spectral Entropy
        -6:         Spectral Flux
        -7:         Spectral Rolloff
        -8-20:      MFCCs
        -21-32:     Chroma Vector
        -33:        Chroma Deviation
    ARGUMENTS:
        - audio:    the input audio signal
    RETURNS:
        - features: an array of features (numOfFeatures x numOfWindows)
    """

    [sample_rate, signal] = aIO.readAudioFile(audio)
    features = aFE.stFeatureExtraction(
        signal,
        sample_rate,
        0.10 * sample_rate,
        0.10 * sample_rate,
    )

    return features


if __name__ == '__main__':
    from os import listdir
    from os.path import (
        isfile,
        join,
        abspath,
        dirname,
    )

    module_directory = abspath(dirname(__file__))

    audio_dir = join(module_directory, "./data/Voices/")

    csv_dir = join(module_directory, "./data/Inputs.csv")

    audio_files = [join(audio_dir, f)
                   for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

    for file in audio_files:
        features = feature_extraction(file)
        print "File:{}\tFeatures:{}".format(file, features)
