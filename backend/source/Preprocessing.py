from pyAudioAnalysis import (
    audioBasicIO as aIO,
    audioSegmentation as aS,
)


class Preprocessing:
    """
    Signal preprocessing:
        Silence remover
    =========================
    Roent Jogno AY 2017 - 2018
    """

    def silence_removal(self, audio):
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
            plot=self.plot
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

    audio_dir = join(abspath(dirname(__file__)), '/data/Voices/')

    audio_files = [join(audio_dir, f)
                   for f in listdir(audio_dir) if isfile(join(audio_dir, f))]

    PP = Preprocessing()

    for file in audio_files:
        segments = PP.silence_removal(file)
        print 'File: {}\tSegments: {}'.format(file, segments)
