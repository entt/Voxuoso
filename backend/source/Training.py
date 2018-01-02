"""
    Machine Learning:
        - Training
    =========================
    (c) Roent Jogno AY 2017 - 2018
"""
from keras import optimizers

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.recurrent import LSTM

from numpy import array, pad
from pandas import read_csv


def create_sequence(input, output):
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

    input_df = read_csv(INPUT_CSV)
    output_df = read_csv(OUTPUT_CSV)

    for index, row in input_df.iterrows():
        mfcc_sequence = []
        for MFCC in headers:
            mfcc_arr = array(row[MFCC].strip('[]').split())
            mfcc_sequence.append(mfcc_arr.astype(float))
        input_sequence.append(mfcc_sequence)

    for index, row in output_df.iterrows():
        output_sequence.append(array(row['avqi_result']))

    for index, item in enumerate(input_sequence):
        for f_index, feature in enumerate(item):
            feature = pad(feature,
                (0, FEATURE_LEN - len(feature)),
                'constant',
                constant_values = (0, feature[-1])
            )
            item[f_index] = feature
        input_sequence[index] = item

    return input_sequence, output_sequence


def build_model():
    """
    Builds neural network model.
    RETURNS:
        - model:    neural network model
    """
    model = Sequential()
    model.add(LSTM(
        13,
        return_sequences=True,
        input_shape=(INPUT_DIM, FEATURE_LEN))
    )
    model.add(LSTM(13, return_sequences=True))
    model.add(LSTM(13))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=optimizers.SGD(lr=0.1),
        loss='mean_squared_error',
        metrics=['accuracy']
    )

    return model


def train_model(model, input, output, epochs, split, model_name='Model'):
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
    model.fit(
        array(input),
        array(output),
        epochs=epochs,
        validation_split=split,
    )

    model.save('./data/{}.h5'.format(model_name))


if __name__ == '__main__':
    INPUT_CSV = "./data/Inputs.csv"
    OUTPUT_CSV = "./data/Outputs.csv"

    FEATURE_LEN = 100
    INPUT_DIM = 13
    OUTPUT_DIM = 1

    headers = [
        'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4',
        'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8',
        'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
        'MFCC13'
    ]

    model = build_model()

    input_sequence, output_sequence = create_sequence(INPUT_CSV, OUTPUT_CSV)

    train_model(model, input_sequence, output_sequence, 10000, 0.2)
