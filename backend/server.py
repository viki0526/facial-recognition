from flask import Flask, request
from flask_cors import CORS
import datetime
from model import ImageClassifier
import logging

x = datetime.datetime.now()

# Initializing flask app
app = Flask(__name__)
CORS(app)

# Route for checking image 
@app.route('/image', methods=['POST'])
def check_image():
    file = request.files.get('image')
    if file:
        file.save('cat_dog_image.jpg')
        classifier = ImageClassifier()
        result = classifier.classifyImage('cat_dog_image.jpg')
        return {'message': 'Image received and processed successfully', 'result': result}, 200
    else:
        return {'error': 'No image file received'}, 400


# Running app
if __name__ == '__main__':
    app.run(debug=True)