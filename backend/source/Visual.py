import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import csv

from keras.models import load_model
from pandas import read_csv
from numpy import (
    array,
    sum,
    genfromtxt,
    zeros_like,
    triu_indices_from,
    bool,
    pad,
    unique,
    linspace,
    interp
)

from os.path import (
    join,
    dirname,
    abspath,
)


class Visual:
    """
    Machine Learning:
        - Plot Loss History
        - Correlation Heat Map
        - Computational Model Scores
    =========================
        Roent Jogno AY 2017 - 2018
    """
    def __init__(self):
        self.input_csv = join(abspath(dirname(__file__)), './data/Inputs.csv')
        self.output_csv = join(abspath(dirname(__file__)), './data/Outputs.csv')
        self.correlation_csv = join(abspath(dirname(__file__)), './data/Correlation.csv')
        self.correlation_matrix = join(abspath(dirname(__file__)), './data/CorrelationMatrix.csv')
        self.prediction_csv = join(abspath(dirname(__file__)), './data/Prediction.csv')

        self.mlp_history = join(abspath(dirname(__file__)), './data/MLP.log')
        self.drnn_history = join(abspath(dirname(__file__)), './data/DRNN.log')

        self.headers = [
            'ZCR', 'EnergyEntropy',
            'SpectralCentroid', 'SpectralSpread',
            'SpectralEntropy', 'SpectralFlux'
        ]

    def training_history(self):
        mlp_hist = genfromtxt(self.mlp_history, delimiter=',', names=True)
        drnn_hist = genfromtxt(self.drnn_history, delimiter=',', names=True)

        figure = plt.figure()
        figure.suptitle('MSE and MAE of both Computational Models')

        mlp_mse = plt.subplot('221')
        mlp_mse.set_title('MLP MSE')
        mlp_mse.plot(mlp_hist['loss'])
        mlp_mse.plot(mlp_hist['val_loss'])

        drnn_mse = plt.subplot('222')
        drnn_mse.set_title('DRNN MSE')
        drnn_mse.plot(drnn_hist['loss'])
        drnn_mse.plot(drnn_hist['val_loss'])

        mlp_mae = plt.subplot('223')
        mlp_mae.set_title('MLP MAE')
        mlp_mae.plot(mlp_hist['mean_absolute_error'])
        mlp_mae.plot(mlp_hist['val_mean_absolute_error'])

        drnn_mae = plt.subplot('224')
        drnn_mae.set_title('DRNN MAE')
        drnn_mae.plot(drnn_hist['mean_absolute_error'])
        drnn_mae.plot(drnn_hist['val_mean_absolute_error'])

        plt.show()

    def correlation(self):
        corr_df = read_csv(self.correlation_csv)

        for feature in self.headers:
            row_values = []
            for index, row in enumerate(corr_df[feature]):
                row_as_array = array(str(row).strip('[]').split()).astype(float)
                row_mean = sum(row_as_array) / len(corr_df[feature])
                row_values.append(row_mean)
            corr_df[feature] = row_values

        correlation = corr_df.corr()
        correlation.to_csv(join(abspath(dirname(__file__)), './data/CorrelationMatrix.csv'))

        mask = zeros_like(correlation, dtype=bool)
        mask[triu_indices_from(mask)] = True

        cmap = sns.cubehelix_palette(8)

        sns.heatmap(
            correlation,
            mask=mask,
            cmap=cmap,
            annot=True
        )

        plt.show()

    def test_results(self):
        output_df = read_csv(self.output_csv)
        input_df = read_csv(self.input_csv)

        input_sequence = []
        output_sequence = []

        mlp_predictions = []
        drnn_predictions = []

        prediction_headers = [
            'file_name', 'actual_result',
            'mlp_prediction', 'drnn_prediction'
        ]

        mlp_model = load_model(join(abspath(dirname(__file__)), 'data/MLP.h5'))
        drnn_model = load_model(join(abspath(dirname(__file__)), 'data/DRNN.h5'))

        for index, row in input_df.iterrows():
            parameter_sequence = []
            for parameters in self.headers:
                parameter_array = array(row[parameters].strip('[]').split())
                parameter_sequence.append(parameter_array.astype(float))
            input_sequence.append(parameter_sequence)

        for index, item in enumerate(input_sequence):
            for f_index, feature in enumerate(item):
                feature = pad(
                    feature,
                    (0, 100 - len(feature)),
                    'constant',
                    constant_values=(0, feature[-1])
                )
                item[f_index] = feature
            input_sequence[index] = item

        for index, row in output_df.iterrows():
            output_sequence.append(array(row['avqi_result']))

        unique_output = unique(output_sequence)
        boundaries = linspace(0, 1.0, num=len(unique_output))
        interpolated = interp(output_sequence, unique_output, boundaries)

        for item in input_sequence:
            item = array(item).reshape((1, 6, 100))
            mlp_predictions.append(str(mlp_model.predict(item)).strip('[]'))
            drnn_predictions.append(str(drnn_model.predict(item)).strip('[]'))

        with open(self.prediction_csv, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=prediction_headers)
            writer.writeheader()

            for index, row in output_df.iterrows():
                writer.writerow({
                    'file_name': row['file_name'],
                    'actual_result': interpolated[index],
                    'mlp_prediction': mlp_predictions[index],
                    'drnn_prediction': drnn_predictions[index],
                })


if __name__ == '__main__':
    vis = Visual()

    parser = argparse.ArgumentParser(description='Data visualization.')
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    if args.mode == 'hist':
        vis.training_history()
    elif args.mode == 'corr':
        vis.correlation()
    elif args.mode == 'results':
        vis.test_results()
