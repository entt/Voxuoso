from flask import (
    Flask,
    request,
    jsonify,
)

app = Flask(__name__)


@app.route('/')
def start():
    return 'Voxuoso API Main Page'


@app.route('/api/v1/voice', methods=['POST'])
def post_voice():
    try:
        data = request.get_json()
        return jsonify({'data': data})  
    except Exception:
        return jsonify({'status': 'ERROR'})


if __name__ == '__main__':
    app.run(port=5000, host="0.0.0.0", use_reloader=True, debug=True)
