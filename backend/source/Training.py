"""
    Machine Learning:
        - Training
    =========================
    (c) Roent Jogno AY 2017 - 2018
"""
import numpy as np
import pandas as pd

INPUT_CSV = "./data/Inputs.csv"
OUTPUT_CSV = "./data/Outputs.csv"


def create_sequence(input=INPUT_CSV, output=OUTPUT_CSV):
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

    input_df = pd.read_csv(input)

    headers = [
        'MFCC1', 'MFCC2', 'MFCC3', 'MFCC4',
        'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8',
        'MFCC9', 'MFCC10', 'MFCC11', 'MFCC12',
        'MFCC13'
    ]

    for n in range(len(input_df)):
        mfcc_sequence = []
        for MFCC in headers:
            i = 0
            mfcc_arr = np.array(input_df[MFCC][i].strip('[]').split())
            i += 1
            mfcc_sequence.append(mfcc_arr)
        input_sequence.append(mfcc_sequence)

    output_df = pd.read_csv(output)

    output_sequence = np.array(output_df['avqi_result'])

    return input_sequence, output_sequence


def build_model():
    pass


def create_test_data(input, output):
    pass


if __name__ == '__main__':
