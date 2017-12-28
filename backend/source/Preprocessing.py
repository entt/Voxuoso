"""
    Signal preprocessing:
        Silence remover
    =========================
    (c) Roent Jogno AY 2017 - 2018
"""
from pyAudioAnalysis import (
    audioBasicIO as aIO,
    audioSegmentation as aS,
)


def silence_removal(audio):
    """
    Divides audio file into voiced-segments.
    ARGUMENTS:
        - audio:        the input audio signal
    RETURNS:
        - segments:     list of segments limits in seconds
    """

    [sample_rate, signal] = aIO.readAudioFile(audio)
    segments = aS.silenceRemoval(
        signal,
        sample_rate,
        0.025, 0.025,
        smoothWindow=1.0,
        Weight=0.6,
        plot=False
    )

    return segments
