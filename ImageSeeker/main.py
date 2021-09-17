'''
@Author: Bappy Ahmed
Date: 03 sep 2021
Email: entbappy73@gmail.com
'''

from flask import Flask, render_template, request,jsonify
from ImageSeeker.com_in_ineuron_ai_utils.utils import decodeImage
from flask_cors import CORS, cross_origin
from ImageSeeker.predict import dogcat
import json
import webbrowser
from threading import Timer



class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = dogcat(self.filename)



app = Flask(__name__) # initializing a flask app
CORS(app)

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")

@app.route('/input',methods=['GET'])  # route to display the home page
def input_form():
    return render_template("input_form.html")

@app.route('/train',methods=['POST','GET']) # route to show the predictions in a web UI
def train_func():
    if request.method == 'POST':
        try:
            # Data config
            TRAIN_DATA_DIR = request.form['TRAIN_DATA_DIR']
            VALID_DATA_DIR = request.form['VALID_DATA_DIR']
            CLASSES =int(request.form['CLASSES'])
            IMAGE_SIZE = request.form['IMAGE_SIZE']
            AUGMENTATION = request.form['AUGMENTATION']
            BATCH_SIZE = int(request.form['BATCH_SIZE'])


            # Model config
            MODEL_OBJ = request.form['MODEL_OBJ']
            MODEL_NAME = request.form['MODEL_NAME']
            FREEZE_ALL = request.form['FREEZE_ALL']
            OPTIMIZER = request.form['OPTIMIZER']
            EPOCHS = int(request.form['EPOCHS'])
            LOSS_FUNC = request.form['LOSS_FUNC']

            configs = {
                "TRAIN_DATA_DIR": TRAIN_DATA_DIR,
                "VALID_DATA_DIR": VALID_DATA_DIR,
                "AUGMENTATION": AUGMENTATION,
                "CLASSES": CLASSES,
                "IMAGE_SIZE": IMAGE_SIZE,
                "BATCH_SIZE": BATCH_SIZE,
                "MODEL_OBJ": MODEL_OBJ,
                "MODEL_NAME": MODEL_NAME,
                "EPOCHS": EPOCHS,
                "FREEZE_ALL": FREEZE_ALL,
                "OPTIMIZER": OPTIMIZER,
                "LOSS_FUNC": LOSS_FUNC

            }

            with open('config.json', 'w') as json_file:
                json.dump(configs, json_file)


            from ImageSeeker import train_engine

            hist = train_engine.train()

            return render_template('input_form.html', output = hist)

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


@app.route('/test',methods=['GET','POST'])  # route to display the home page
def predcit():
    return render_template("predict.html")


@app.route("/predict", methods=['POST'])
def predictRoute():
    clApp = ClientApp()
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.predictiondogcat()
    return jsonify(result)



def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


def start_app():
    Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=8080,debug=True)



if __name__ == "__main__":
    start_app()
