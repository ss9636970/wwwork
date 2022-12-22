import traceback
import flask
from flask import jsonify, request
import pandas as pd
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = True  # 正式環境關掉
app.config["JSON_AS_ASCII"] = False


@app.route('/predict', methods=['POST'])
def country():
    try:
        data = {'success':False}
        if request.form.get('image'):
            data['image'] = request.form.get('image')
            return jsonify(data)

        else:
            return jsonify('fail')

    except Exception as e:
        return traceback.format_exc()
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=20)