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
        0.05, 0.05,
        smoothWindow=1.0,
        Weight=0.6,
        plot=False
    )

    return segments


if __name__ == '__main__':
    from os import listdir
    from os.path import (
        isfile,
        join,
        abspath,
        dirname,
    )

    module_directory = abspath(dirname(__file__))

    audio_dir = join(module_directory, './data/Voices/')

    csv_dir = join(module_directory, './data/Inputs.csv')

    audio_files = [join(audio_dir, f)
                   for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

    for file in audio_files:
        segments = silence_removal(file)
        print 'File: {}\tSegments: {}'.format(file, segments)
