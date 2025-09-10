from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# yolo model for queen cell version 2
model = YOLO('queen-cell-2.pt', task='detect', device='cpu', weights_only=False)

# with image extensions
# allow only image extensions
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "flask yolov8 server is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "no image uploaded"}), 400

    file = request.files["image"]

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "invalid file type"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        # add confidence filter (example: 0.3 threshold)
        results = model.predict(img, conf=0.3)

        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "xmin": float(box.xyxy[0][0]),
                    "ymin": float(box.xyxy[0][1]),
                    "xmax": float(box.xyxy[0][2]),
                    "ymax": float(box.xyxy[0][3]),
                    "confidence": round(float(box.conf[0]), 3),
                    "name": model.names[int(box.cls[0])]
                })

        return jsonify({"success": True, "detections": detections})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# displaying
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
