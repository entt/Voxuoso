from flask import (
    Flask,
    request,
    jsonify,
)
from os.path import (
    join,
    dirname,
    abspath,
)

UPLOAD_FOLDER = join(abspath(dirname(__file__)), 'received/')
ALLOWED_EXTENSIONS = set(['wav', 'aiff', 'mp3', 'mp4'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def start():
    return 'Voxuoso API Main Page'


@app.route('/api/v1/voice', methods=['GET', 'POST'])
def post_voice():
    if request.method == 'POST':
        try:
            data = request.get_json()
            return jsonify({'data': data})
        except Exception:
            return jsonify({'status': 'ERROR'})


if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", use_reloader=True, debug=True)
