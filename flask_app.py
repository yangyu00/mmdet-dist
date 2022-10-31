# -*- encoding: utf-8 -*-

import flask
from web.behavior import behavior_guide

app = flask.Flask(__name__)

app.register_blueprint(behavior_guide)

if __name__ == "__main__":

    app.run(host='192.168.169.2', port=8888, debug=True, use_reloader=False)
