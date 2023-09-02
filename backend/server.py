from flask import Flask
import datetime
from model import ImageClassifier
import logging

x = datetime.datetime.now()

# Initializing flask app
app = Flask(__name__)


# Route for seeing a data
@app.route('/data')
def get_time():
    # Returning an api for showing in reactjs
    return {
        'Name':"Vikrant Prakash",
        "Age":"21",
        "Date":x,
        "programming":"Python Flask"
    }
# Route for posting image 
@app.route('/image')
def set_image(filePath):
    logging.debug(f'The filePath is: {filePath}')
    return True


# Running app
if __name__ == '__main__':
    app.run(debug=True)