import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request, jsonify, Flask
import h5py
import tensorflow as tf
app = Flask(__name__)

def get_model():
    global model
    model = load_model('VGG16_cats_and_dogs.h5')
    model._make_predict_function()
    print(" * Model Loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model..")
get_model()
global graph
graph = tf.compat.v1.get_default_graph()

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)

    dataBytesIO = io.BytesIO(decoded)

    image = Image.open(dataBytesIO)
    processed_image = preprocess_image(image, target_size=(224, 224))

    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'dog': prediction[0][0],
            'cat': prediction[0][1]
        }
    }
    return jsonify(response)
