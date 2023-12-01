import flask
# app.py

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from vit_model import predict_vit
from yolo_model import predict_yolo

app = Flask(__name__, static_folder='static/uploads')
app.config['UPLOAD_FOLDER'] = 'static' + '\'+uploads'

# Brief descriptions of the models
vit_description = "Vision Transformer Model: Predicts objects in images and returns the predicted object as text."
yolo_description = "YOLO Model: Predicts objects in images and returns an image with bounding boxes drawn."

@app.route('/')
def index():
    return render_template('index.html', vit_description=vit_description, yolo_description=yolo_description)

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = request.form['model']
    file = request.files['file']

    if file:
        filename = secure_filename(file.filename)
        file_path = f"{filename}"
        file.save(file_path)

        if selected_model == 'vit':
            prediction = predict_vit(file_path)
            return render_template('vit_prediction.html', prediction=prediction)

        elif selected_model == 'yolo':
            prediction_image_path = predict_yolo(file_path)
            print(type(prediction_image_path))
            return render_template('yolo_prediction.html', image_path=prediction_image_path)

    return "Prediction failed."

if __name__ == '__main__':
    app.run(debug=True)
