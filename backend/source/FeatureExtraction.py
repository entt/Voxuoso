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
<<<<<<< HEAD
        - audio:    the input audio signal
    RETURNS:
        - features: an array of features (numOfFeatures x numOfWindows)
=======
        - audio:        the input audio signal
    RETURNS:
        - features:     an array of features (numOfFeatures x numOfShortTermWindows)
>>>>>>> 8aeddc75eb0cf8c1ebc1d761a8688d9aa71cad5d
    """

    [sample_rate, signal] = aIO.readAudioFile(audio)
    features = aFE.stFeatureExtraction(
        signal,
        sample_rate,
        0.10 * sample_rate,
        0.10 * sample_rate,
    )

    return features
