from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM

from pandas import read_csv
from numpy import (
    array,
    pad,
    linspace,
    interp,
    unique
)
from matplotlib import pyplot as plt

from math import floor
from os.path import (
    join,
    dirname,
    abspath,
)
from time import time


class Train:
    """
    Machine Learning:
        - Training
    =========================
    Roent Jogno AY 2017 - 2018
    """

    def __init__(self):
        self.input_csv = join(abspath(dirname(__file__)), './data/Inputs.csv')
        self.output_csv = join(abspath(dirname(__file__)), './data/Outputs.csv')
        
        self.feature_length = 100
        self.input_dim = 13
        self.output_dim = 1

        self.headers = [
            'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4',
            'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8',
            'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
            'MFCC13'
        ]

        self.training_time = 0

    def create_sequence(self, input, output):
        """
        Creates sequences from CSV files.
        ARGUMENTS:
            - input:            input csv file
            - output:           output csv file
        RETURNS:
            - input_sequence:   a list of lists containing features
            - output_sequence:  a list of AVQI result
        """
        input_sequence = []
        output_sequence = []

        input_df = read_csv(self.input_csv)
        output_df = read_csv(self.output_csv)

        for index, row in input_df.iterrows():
            mfcc_sequence = []
            for MFCC in self.headers:
                mfcc_arr = array(row[MFCC].strip('[]').split())
                mfcc_sequence.append(mfcc_arr.astype(float))
            input_sequence.append(mfcc_sequence)

        for index, row in output_df.iterrows():
            output_sequence.append(array(row['avqi_result']))

        for index, item in enumerate(input_sequence):
            for f_index, feature in enumerate(item):
                feature = pad(feature,
                    (0, self.feature_length - len(feature)),
                    'constant',
                    constant_values = (0, feature[-1])
                )
                item[f_index] = feature
            input_sequence[index] = item

        unique_output = unique(output_sequence)
        boundaries = linspace(0, 1.0, num=len(unique_output))
        interpolated = interp(output_sequence, unique_output, boundaries)

        for index, item in enumerate(interpolated):
            output_sequence[index] = array(item)

        return input_sequence, output_sequence


    def build_model(self):
        """
        Builds neural network model.
        RETURNS:
            - model:    neural network model
        """
        model = Sequential()
        model.add(LSTM(64,
            dropout=0.20,
            recurrent_dropout=0.35,
            activation='sigmoid',
            input_shape=(self.input_dim, self.feature_length)
        ))
        model.add(Dense(self.output_dim, activation='sigmoid'))

        model.compile(
            optimizer=optimizers.SGD(lr=0.01),
            loss='mse',
            metrics=['mae', 'logcosh']
        )

        return model


    def train_model(self, model, input, output, epochs, split=0, model_name='Model'):
        """
        Trains the model and saves it in HDF5 format.
        ARGUMENTS:
            - model:        neural network model
            - input:        input sequence
            - output:       output sequence
            - epochs:       num of epochs
            - split:        percentage of training and test split
            - model_name:   name of model to be saved
        """
        start_training = time()
        history = model.fit(
            array(input),
            array(output),
            epochs=epochs,
            validation_split=split
        )
        end_training = time()

        self.training_time = end_training - start_training

        model.save(join(abspath(dirname(__file__)), './data/{}.h5'.format(model_name)))

        return history, self.training_time


    def test_model(self, model, input_seq, output_seq):
        x_test = input_seq[int(floor(len(input_seq) * 0.8)):]
        y_test = output_seq[int(floor(len(output_seq) * 0.8)):]

        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test score: ", score[0])
        print("Test accuracy: ", score[1])


if __name__ == '__main__':
    train = Train()

    model = train.build_model()

    input_sequence, output_sequence = train.create_sequence(train.input_csv, train.output_csv)

    history, time = train.train_model(model, input_sequence, output_sequence, 10000)

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['mean_absolute_error'])
    # plt.show()

    print "Training time: {}".format(time)
