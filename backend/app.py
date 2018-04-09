from flask import (
    Flask,
    request,
    jsonify
)
from os.path import (
    join,
    dirname,
    abspath
)

from numpy import (
    array,
    pad
)

from keras.models import load_model
from pyAudioAnalysis import (
    audioFeatureExtraction as aFE,
    audioBasicIO as aIO
)

app = Flask(__name__)

sample_rate = 44100
timesteps = 100
input_dim = 6


@app.route('/')
def start():
    return 'Voxuoso API Main Page'


@app.route('/api/v1/voice')
def post_voice():
    voice = request.files['sound']
    is_drnn = request.data.get('isDRNN')

    if is_drnn is True:
        model = load_model(join(abspath(dirname(__file__)), 'data/DRNN.h5'))
    elif is_drnn is False:
        model = load_model(join(abspath(dirname(__file__)), 'data/MLP.h5'))

    [sample_rate, signal] = aIO.readAudioFile(voice)

    features = aFE.stFeatureExtraction(
        signal,
        sample_rate,
        0.05 * sample_rate,
        0.05 * sample_rate
    )

    input_data = [
        features[0], features[2],
        features[3], features[4],
        features[5], features[6],
    ]

    for index, feature in enumerate(input_data):
        feature = pad(
            feature,
            (0, timesteps - len(feature)),
            'constant',
            constant_values=(0, feature[-1])
        )
        input_data[index] = feature

    input_data = array(input_data).reshape((1, input_dim, timesteps))
    prediction = model.predict(input_data)

    prediction = str(prediction * 100).strip('[]')

    response = {
        'output': prediction
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", use_reloader=True, debug=True)
