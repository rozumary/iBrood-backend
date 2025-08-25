from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow all origins


# Load your YOLOv8 model
model = YOLO('queen-cell-2.pt')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')

    results = model.predict(img)

    data = []
    for r in results:
        for box in r.boxes:
            data.append({
                'xmin': box.xyxy[0][0].item(),
                'ymin': box.xyxy[0][1].item(),
                'xmax': box.xyxy[0][2].item(),
                'ymax': box.xyxy[0][3].item(),
                'confidence': box.conf[0].item(),
                'name': model.names[int(box.cls[0].item())]
            })
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

@app.route('/')
def index():
    return "Flask server is running!"

