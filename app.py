from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins

# Load YOLO model once (not inside function, so it's faster)
model = YOLO('queen-cell-2.pt', task='detect', device='cpu', weights_only=False)


@app.route("/", methods=["GET"])
def index():
    return "Flask YOLOv8 server is running!"


@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image is uploaded
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Open image
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        results = model.predict(img)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "xmin": float(box.xyxy[0][0]),
                    "ymin": float(box.xyxy[0][1]),
                    "xmax": float(box.xyxy[0][2]),
                    "ymax": float(box.xyxy[0][3]),
                    "confidence": float(box.conf[0]),
                    "name": model.names[int(box.cls[0])]
                })

        return jsonify(detections)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

