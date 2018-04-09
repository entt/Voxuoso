from Tkinter import *
import tkMessageBox
import tkFileDialog

from pyAudioAnalysis import audioFeatureExtraction as aFE

from scipy.io import wavfile
from numpy import (
    array,
    pad
)

from keras.models import load_model
from os.path import (
    join,
    dirname,
    abspath,
)

from PIL import Image, ImageTk
import sounddevice as sd


class VoxGUI:
    def __init__(self, master):
        self.master = master
        self.duration = 2
        self.samplerate = 44100
        self.input_dim = 6
        self.timesteps = 100
        self.file_name = ''
        self.mode = IntVar()

        master.title('Voxuoso')

        # self.record = Image.open(join(abspath(dirname(__file__)), 'assets/record.png'))
        # self.record_photo = ImageTk.PhotoImage(self.record)
        self.RecordButton = Button(
            master,
            text='Record',
            # image=self.record_photo,
            # compound=LEFT,
            command=self.record,
        ).grid(
            row=0, column=0,
            columnspan=2, sticky=W + E
        )

        self.icon = Image.open(join(abspath(dirname(__file__)), 'assets/icon.png'))
        self.icon_photo = ImageTk.PhotoImage(self.icon)
        self.IconLabel = Label(
            image=self.icon_photo
        ).grid(
            row=0, column=1,
            rowspan=2, columnspan=2,
            sticky=E
        )

        # self.play = Image.open(join(abspath(dirname(__file__)), 'assets/play.png'))
        # self.play_photo = ImageTk.PhotoImage(self.play)
        self.PlayButton = Button(
            master,
            text='Play',
            # image=self.play_photo,
            # compound=LEFT,
            command=self.play,
        ).grid(row=1, column=0, sticky=W)

        # self.stop = Image.open(join(abspath(dirname(__file__)), 'assets/stop.png'))
        # self.stop_photo = ImageTk.PhotoImage(self.stop)
        self.StopButton = Button(
            master,
            text='Stop',
            # image=self.stop_photo,
            # compound=LEFT,
            command=self.stop,
        ).grid(row=1, column=1, sticky=E)

        self.MLPRadio = Radiobutton(
            master,
            text='MLP',
            variable=self.mode,
            value=1,
        ).grid(row=2, column=0)

        self.DRNNRadio = Radiobutton(
            master,
            text='DRNN',
            variable=self.mode,
            value=2,
        ).grid(row=2, column=1)

        self.OpenButton = Button(
            master,
            text='Open Sound File',
            command=self.open,
        ).grid(row=2, column=2)

        self.PredictButton = Button(
            master,
            text='Predict',
            command=self.predict,
        ).grid(row=2, column=3)

    def record(self):
        print('Recording for 2 seconds.')
        self.recording = sd.rec(
            int(self.duration * self.samplerate),
            samplerate=self.samplerate,
            channels=1
        )
        print('Done recording.')

    def open(self):
        self.file_name = tkFileDialog.askopenfilename(
            initialdir=dirname(__file__),
            title="Select audio file",
            filetypes=(("WAV files", "*.wav"), ("all files", "*.*"))
        )

        self.samplerate, self.recording = wavfile.read(self.file_name)

    def play(self):
        print('Playing for 2 seconds.')
        sd.play(self.recording)

    def stop(self):
        print('Stopped playing audio.')
        sd.stop()

    def predict(self):
        try:
            if self.mode.get() == 1:
                print('Predicting using MLP...')
                model = load_model(join(abspath(dirname(__file__)), 'data/MLP.h5'))
            else:
                print('Predicting using DRNN...')
                model = load_model(join(abspath(dirname(__file__)), 'data/DRNN.h5'))

            features = aFE.stFeatureExtraction(
                self.recording,
                self.samplerate,
                0.05 * self.samplerate,
                0.05 * self.samplerate
            )

            input_data = [
                features[0], features[2],
                features[3], features[4],
                features[5], features[6],
            ]

            for index, feature in enumerate(input_data):
                feature = pad(
                    feature,
                    (0, self.timesteps - len(feature)),
                    'constant',
                    constant_values=(0, feature[-1])
                )
                input_data[index] = feature

            input_data = array(input_data).reshape((1, self.input_dim, self.timesteps))
            prediction = model.predict(input_data)
            prediction = str(prediction * 100).strip('[]')

            if float(prediction) <= 33:
                severity = 'Mildly Deviant (MI)'
            elif 34 <= float(prediction) <= 66:
                severity = 'Moderately Deviant (MO)'
            elif float(prediction) >= 67:
                severity = 'Severely Deviant (SE)'

            tkMessageBox.showinfo('Info', 'Severity of Layngitis: {}\n{}'.format(prediction, severity))
        except AttributeError:
            tkMessageBox.showerror('Error', 'No audio file for processing.')


root = Tk()
vox = VoxGUI(root)
root.mainloop()
