from flask import Flask, jsonify, request, send_file
import os
import time
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
app = Flask(__name__)

#load model 
learn = None

#function that loads model named export.pkl
#export.pkl is in same directory as this python file
def load_model():
    global learn
    learn = load_learner('models/')

'''
fn that receives post request, and downloads incoming file
High resolution file is generated from the model, and saved.
The filepath is sent back to client
'''
@app.route("/", methods=['POST'])
def predict():
    keylist = ''
    for x in request.files:
        keylist += x
        keylist += ','
    try:
        print(request.headers)
        data = request.files['file']
    except:
        print("error: wrong key" + "\n" + keylist)
        return jsonify({'message':'invalid key','current keylist':keylist})
    
    filename = 'Uploads/uploaded_img-{}.png'.format(int(time.time()))
    data.save(filename)
    image = open_image(filename)
    _,img_hr,b = learn.predict(image)
    hr_filename = 'hi-res-{}.png'.format(int(time.time()))
    hr_savepath = 'Predicted/' + hr_filename
    Image(img_hr).save(hr_savepath)
    hr_filename = '%' + hr_filename + '%'
    return jsonify({"filename":hr_filename,"status":"ok"})

'''
function that sends file to client
the request from client will come with a (str)filename.
The image in the corresponding filename is returned to client
curl command:
curl http://6fd9b167.ngrok.io/download?filename="hi-res.png" -o from-server.png
'''
@app.route("/download", methods=['GET'])
def return_image():
    filepath = 'Predicted/' + request.args.get('filename')
    print(filepath)
    try:
        if os.path.isfile(filepath):
            return send_file(filepath)
        else:
            print("file not found")
            return jsonify({'message':'file not found'})
    except Exception as e:
        return jsonify({'message':'error encountered','error message':str(e)})

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    load_model()
    print("model_loaded")
    app.run()