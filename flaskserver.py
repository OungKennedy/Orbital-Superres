from flask import Flask, jsonify, request, send_file
import os
import time
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
import torchvision
app = Flask(__name__)

#load model 
learn = None

#function that loads model named export.pkl
#export.pkl is in same directory as this python file
def load_model():
    global learn
    learn = load_learner('models/superres_full_0706')
def round_up_to_even(num):
    if int(num) % 2 == 0:
        return int(num)
    else:
        return int(num)+1
def get_resize(y, z, max_size):
    if y*2 <= max_size and z*2 <= max_size:
        y_new = y*2
        z_new = z*2
    else:
        if y > z:
            y_new = max_size
            z_new = int(round_up_to_even(z * max_size / y))

        else:
            z_new = max_size
            y_new = int(round_up_to_even(y * max_size / z))
    return (y_new, z_new)
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
    input_x,input_y,input_z = image.shape
    output_y, output_z = get_resize(input_y,input_z,1000)
    databunch = (ImageImageList.from_folder('Predicted').split_none()
          .label_from_func(lambda x: x)
          .transform(get_transforms(), size=(output_y, output_z), tfm_y=True)
          .databunch(bs=2, no_check=True).normalize(imagenet_stats, do_y=True))
    databunch.c = 3
    learn.data = databunch      
    _,img_hr,b = learn.predict(image)
    img_hr = img_hr.clamp(0,1) 
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
    app.run(host='0.0.0.0', host = 80)
