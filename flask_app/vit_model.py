from transformers import ViTImageProcessor, ViTForImageClassification # ViTFeatureExtractor
from PIL import Image

import warnings
warnings.filterwarnings('ignore')


def predict_vit(image): 
    img = Image.open(image)

    # Instantiate the feature extractor specific to the model checkpoint
    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # Instantiate the pretrained model
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    # Extract features (patches) from the image
    inputs = feature_extractor(images=img, return_tensors="pt")

    # Predict by feeding the model (** is a python operator which unpacks the inputs)
    outputs = model(**inputs)

    # Convert outputs to logis
    logits = outputs.logits

    # model predicts one of the classes by pick the logit which has the highest probability
    predicted_class_idx = logits.argmax(-1).item()

    return model.config.id2label[predicted_class_idx]
