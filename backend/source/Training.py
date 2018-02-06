from keras import optimizers
from keras.models import Sequential
from keras.layers import (
    Dense,
    Flatten,
    Dropout
)
from keras.layers.recurrent import LSTM
from keras.callbacks import CSVLogger

from pandas import read_csv
from numpy import (
    array,
    pad,
    linspace,
    interp,
    unique,
    genfromtxt
)
import matplotlib.pyplot as plt

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
        - Data Preparation
        - Ouput Interpolation
        - Training
        - Plot Loss History
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

        print(interpolated)

        for index, item in enumerate(interpolated):
            output_sequence[index] = array(item)

        return input_sequence, output_sequence

    def build_model(self, mode):
        """
        Builds neural network model.
        ARGUMENTS:
            - mode:     either MLP or DRNN
        RETURNS:
            - model:    neural network model
        """
        if mode == 'DRNN':
            model = Sequential([
                LSTM(
                    150,
                    activation='sigmoid',
                    dropout=0.65,
                    recurrent_dropout=0.50,
                    return_sequences=True,
                    input_shape=(self.input_dim, self.feature_length)
                ),
                LSTM(
                    150,
                    activation='sigmoid',
                    dropout=0.65,
                    recurrent_dropout=0.50,
                    input_shape=(self.input_dim, self.feature_length)
                ),
                Dense(
                    self.output_dim,
                    activation='sigmoid'
                ),
            ])

        elif mode == 'MLP':
            model = Sequential([
                Dense(
                    100,
                    activation='sigmoid',
                    input_shape=(self.input_dim, self.feature_length)
                ),
                Dropout(0.65),
                Dense(
                    100,
                    activation='sigmoid',
                    input_shape=(self.input_dim, self.feature_length)
                ),
                Dropout(0.65),
                Dense(
                    100,
                    activation='sigmoid',
                    input_shape=(self.input_dim, self.feature_length)
                ),
                Dropout(0.65),
                Flatten(),
                Dense(
                    self.output_dim,
                    activation='sigmoid'
                ),
            ])

        model.compile(
            optimizer=optimizers.SGD(lr=0.1),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train_model(self, model, training_input, training_output, epochs, split=0, model_name='Model'):
        """
        Trains the model and saves it in HDF5 format.
        ARGUMENTS:
            - model:                neural network model
            - training_input:       training_input sequence
            - training_output:      training_output sequence
            - epochs:               num of epochs
            - split:                percentage of training and test split
            - model_name:           name of model to be saved
        """
        start_training = time()
        csv_logger = CSVLogger(join(abspath(dirname(__file__)), './data/{}.log'.format(model_name)))
        history = model.fit(
            array(training_input),
            array(training_output),
            epochs=epochs,
            batch_size=len(training_output),
            validation_split=split,
            callbacks=[csv_logger]
        )
        end_training = time()

        self.training_time = end_training - start_training

        model.save(join(abspath(dirname(__file__)), './data/{}.h5'.format(model_name)))
        print('Model {}.h5 saved.'.format(model_name))
        print('Log {}.log saved.'.format(model_name))

        return history

    def plot_history(self, history_file, model_name):
        history = genfromtxt(history_file, delimiter=',', names=True)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model Loss ({})'.format(model_name))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()


if __name__ == '__main__':
    model_name = 'DRNN'

    train = Train()
    model = train.build_model('{}'.format(model_name))

    input_sequence, output_sequence = train.create_sequence(train.input_csv, train.output_csv)

    # training_data_X = input_sequence[:int(floor(len(input_sequence) * .60))]
    # training_data_Y = output_sequence[:int(floor(len(output_sequence) * .60))]

    # test_data_X = input_sequence[int(floor(len(input_sequence) * .60)):]
    # test_data_Y = output_sequence[int(floor(len(output_sequence) * .60)):]

    # history = train.train_model(model, (training_data_X), training_data_Y, 1000, 0.2, model_name=model_name)
    # print "Training time: {}".format(train.training_time)

    # score = model.evaluate(array(test_data_X), array(test_data_Y), batch_size=len(test_data_Y), verbose=0)
    # print('Scores:\nMSE: {}\tMAE: {}'.format(*score))

    # train.plot_history(join(abspath(dirname(__file__)), './data/{}.log').format(model_name), model_name)
