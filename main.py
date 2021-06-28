import base64

from flask import Flask
from flask_socketio import SocketIO

import constants

app = Flask(__name__)
app.config['SECRET_KEY'] = 'this_is_secret_123'
socketio = SocketIO(app, cors_allowed_origins='*')


@app.route('/')
def index():
    return 'Hello, world!'


@socketio.on(constants.EVENT_MESSAGE)
def handle_message(message):
    print(message)
    return f'OK, {message}'


@socketio.event()
def image(data):
    with open("received_image.jpg", "wb") as fh:
        fh.write(base64.b64decode(data))

    print(data)


if __name__ == '__main__':
    socketio.run(app, host='192.168.0.21', port=9990, debug=True)
