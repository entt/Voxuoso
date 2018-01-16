from pyAudioAnalysis import (
    audioBasicIO as aIO,
    audioFeatureExtraction as aFE,
)


class FeatureExtraction:
    """
    Signal processing:
        Feature Extraction:
            - MFCC
    =========================
    Roent Jogno AY 2017 - 2018
    """

    def feature_extraction(self, audio):
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
            0.05 * sample_rate,
            0.05 * sample_rate,
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

    audio_dir = join(abspath(dirname(__file__)), '/data/Voices/')

    audio_files = [join(audio_dir, f)
                   for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

    FE = FeatureExtraction()

    for file in audio_files:
        features = FE.feature_extraction(file)
        print 'Feature shape: {}'.format(features.shape)
        print 'File: {}\tFeatures: {}'.format(file, features)
