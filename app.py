from flask import Flask, request, jsonify
from flask_cors import CORS
import model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

leviosaModel = model.LevioasModel()

@app.route('/upload', methods=['POST'])
def upload():
    res = []
    for file in request.files.values():
        image = Image.open(io.BytesIO(file.read()))
        predict = leviosaModel.predict(np.array(image))
        if predict!="":
            res.append(predict)
    return jsonify({"res":res})
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)