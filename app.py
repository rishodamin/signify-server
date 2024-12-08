import base64
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
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']


    try:
        pil_image = Image.open(file)
        res = leviosaModel.predict(numpy_image=np.array(pil_image))

        return jsonify({
            "res": res,
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)