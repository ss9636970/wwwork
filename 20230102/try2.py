import traceback
import flask
from flask import jsonify, request
import pandas as pd
import numpy as np
import cv2

app = flask.Flask(__name__)
app.config["DEBUG"] = True  # 正式環境關掉
app.config["JSON_AS_ASCII"] = False


@app.route('/predict', methods=['POST'])
def country():
    try:
        data = {}
        if request.data:
            png_bytes = request.data

            c1 = np.frombuffer(png_bytes, np.uint8)
            img_frombyte = cv2.imdecode(c1, cv2.IMREAD_COLOR)
            print(img_frombyte.shape)

            c, img_encoded = cv2.imencode('.png', img_frombyte)
            png_bytes = img_encoded.tobytes()

            return jsonify(png_bytes.decode(encoding='ISO-8859-1'))

        elif request.form.get('image'):
            png_bytes = request.form.get('image')
            return jsonify(png_bytes)
            # print(png_bytes)

            c = np.fromstring(png_bytes, np.uint8)
            print('shape c:', c.shape)
            img_frombyte = cv2.imdecode(c, cv2.IMREAD_COLOR)
            print('shape img_frombyte', img_frombyte.shape)

            data['image'] = img_frombyte
            data['success'] = True
            return jsonify(data)

        else:
            return jsonify('fail')

    except Exception as e:
        return traceback.format_exc()
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=20)