from keras import optimizers
from keras.models import (
    Sequential,
    load_model
)
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
    unique
)

from math import floor
from os.path import (
    join,
    dirname,
    abspath,
)
from time import time
import argparse


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

        self.input_sequence, self.output_sequence = self.create_sequence(self.input_csv, self.output_csv)

        self.training_data_X = self.input_sequence[:int(floor(len(self.input_sequence) * .90))]
        self.training_data_Y = self.output_sequence[:int(floor(len(self.output_sequence) * .90))]

        self.test_data_X = self.input_sequence[int(floor(len(self.input_sequence) * .90)):]
        self.test_data_Y = self.output_sequence[int(floor(len(self.output_sequence) * .90)):]

        self.timesteps = 100
        self.training_batch_size = len(self.training_data_Y)
        self.test_batch_size = len(self.test_data_Y)
        self.input_dim = 6
        self.output_dim = 1

        self.headers = [
            'ZCR', 'EnergyEntropy',
            'SpectralCentroid', 'SpectralSpread',
            'SpectralEntropy', 'SpectralFlux'
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

        headers = [
            'ZCR', 'EnergyEntropy',
            'SpectralCentroid', 'SpectralSpread',
            'SpectralEntropy', 'SpectralFlux'
        ]
        timesteps = 100

        input_df = read_csv(self.input_csv)
        output_df = read_csv(self.output_csv)

        for index, row in input_df.iterrows():
            parameter_sequence = []
            for parameters in headers:
                parameter_array = array(row[parameters].strip('[]').split())
                parameter_sequence.append(parameter_array.astype(float))
            input_sequence.append(parameter_sequence)

        for index, row in output_df.iterrows():
            output_sequence.append(array(row['avqi_result']))

        for index, item in enumerate(input_sequence):
            for f_index, feature in enumerate(item):
                feature = pad(
                    feature,
                    (0, timesteps - len(feature)),
                    'constant',
                    constant_values=(0, feature[-1])
                )
                item[f_index] = feature
            input_sequence[index] = item

        unique_output = unique(output_sequence)
        boundaries = linspace(0, 1.0, num=len(unique_output))
        interpolated = interp(output_sequence, unique_output, boundaries)

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
                    dropout=0.25,
                    recurrent_dropout=0.25,
                    return_sequences=True,
                    input_shape=(self.input_dim, self.timesteps)
                ),
                LSTM(
                    150,
                    dropout=0.25,
                    recurrent_dropout=0.25,
                ),
                Dense(
                    self.output_dim,
                    activation='sigmoid'
                ),
            ])

        elif mode == 'MLP':
            model = Sequential([
                Dense(
                    150,
                    input_shape=(self.input_dim, self.timesteps)
                ),
                Dropout(0.25),
                Dense(150),
                Dropout(0.25),
                Flatten(),
                Dense(
                    self.output_dim,
                    activation='sigmoid'
                ),
            ])

        model.compile(
            optimizer=optimizers.SGD(lr=0.001),
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
            batch_size=self.training_batch_size,
            validation_split=0.2,
            callbacks=[csv_logger]
        )
        end_training = time()

        self.training_time = end_training - start_training

        model.save(join(abspath(dirname(__file__)), './data/{}.h5'.format(model_name)))
        print('Model {}.h5 saved.'.format(model_name))
        print('Log {}.log saved.'.format(model_name))

        return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data visualization.')
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    if args.mode == 'MLP':
        model_name = 'MLP'
    elif args.mode == 'DRNN':
        model_name = 'DRNN'

    train = Train()

    # Load existing model
    model = load_model(join(abspath(dirname(__file__)), './data/{}.h5'.format(model_name)))

    # Create model
    # model = train.build_model('{}'.format(model_name))
    # history = train.train_model(model, train.training_data_X, train.training_data_Y, 5000, 0.2, model_name=model_name)
    # print "Training time: {}".format(train.training_time)

    score = model.evaluate(array(train.test_data_X), array(train.test_data_Y), batch_size=train.test_batch_size, verbose=0)
    print('Scores:\nMSE: {}\tMAE: {}'.format(*score))
