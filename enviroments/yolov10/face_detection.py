from ultralytics import YOLOv10
import os
from IPython.display import Image
from PIL import Image

from flask import Flask, request, send_file
from PIL import Image, ImageFilter
import io

import os
import shutil
HOME = os.getcwd()
print(HOME)

path=f"{HOME}/best.pt"
model = YOLOv10(path)
app = Flask(__name__)

@app.route('/process-image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    image = Image.open(file.stream)
    
    results = model.predict(image)
    results_path = f"{HOME}/runs/detect/exp1/"
    os.makedirs(results_path, exist_ok=True)
    results[0].save(os.path.join(results_path, "detect.jpeg"))

    img=Image.open(f"{HOME}/runs/detect/exp1/detect.jpeg")
    processed_image=img

    # Lưu ảnh đã xử lý vào bộ nhớ đệm
    img_io = io.BytesIO()
    processed_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)