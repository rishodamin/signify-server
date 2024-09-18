from flask import Flask, request, jsonify
from flask_cors import CORS
import model
from PIL import Image
import numpy as np
import io
import base64


app = Flask(__name__)
CORS(app)

leviosaModel = model.LevioasModel()

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    res = []
    for file in data['files']:
        image = Image.open(io.BytesIO(base64.b64decode(file)))
        predict = leviosaModel.predict(np.array(image))
        if predict!="":
            res.append(predict)
    return jsonify({"res":res})
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)