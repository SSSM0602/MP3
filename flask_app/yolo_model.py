from ultralytics import YOLO
from PIL import Image

import warnings
warnings.filterwarnings('ignore')

def predict_yolo(image):
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Run inference on 'bus.jpg'
    results = model(image,verbose=False)  # results list

    # Save results as image
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        im.save('flask_app\\static\\uploads\\results.png')  # save image

        return 'results.png'