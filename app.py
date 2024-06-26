from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import io

app = Flask(__name__)

# Load your trained model
model = load_model('models/lenet_5_v2.keras')

def preprocess_image(image_stream):
    # Load the image
    image = Image.open(image_stream)

    # Convert the image to grayscale
    image = image.convert('L')

    # Invert the image (black background and white digits) if necessary
    image = ImageOps.invert(image)

    # Resize the image to fit within a 20x20 box while keeping the aspect ratio
    image.thumbnail((20, 20), Image.ANTIALIAS)

    # Create a new 28x28 white image
    new_image = Image.new('L', (28, 28), (255))

    # Calculate the position to paste the resized image so that it is centered
    upper_left = ((28 - image.width) // 2, (28 - image.height) // 2)
    new_image.paste(image, upper_left)

    return new_image

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/mnist")
def mnist():
    return render_template("mnist.html")

@app.route("/predict", methods=['POST'])
def predict():
    if 'file' in request.files:
        # If an image file is uploaded
        file = request.files['file']
        image = preprocess_image(file.stream)
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)
    else:
        # If pixel data is sent as JSON
        data = request.json
        pixels = np.array(data['pixels'], dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        image_array = pixels

    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    return jsonify({'digit': int(digit)})

if __name__ == "__main__":
    app.run(debug=True)
