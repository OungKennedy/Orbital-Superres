from flask import Flask, jsonify, request, send_file, g
import os
import time

from Scripts.helpers import feature_loss
from fastai.basic_train import load_learner
from fastai.vision import Image
import torchvision
import Scripts.utils
from Scripts.size_change import change_output_size

from werkzeug.utils import secure_filename

from face_detection import LandmarksDetector, image_align
from utils import feature_loss
import __main__
__main__.FeatureLoss = feature_loss

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
save_dir = tempfile.TemporaryDirectory(dir=static)

#load model from models directory
model = None 

def load_model():
    global model
    model = load_learner('models','resnet34-enhance.pkl')
    print('model loaded')

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def generate_faces(img_path, filename):
    landmarks_detector = LandmarksDetector()
    img_paths = []
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(img_path),start=1):
        # Extract face and save at 512px
        img = image_align(img_path, face_landmarks)
        fp_bef = tempfile.NamedTemporaryFile(suffix=str(i)+'bef'+filename, dir=save_dir.name, delete=False)
        img_save(fp_bef,'PNG')

        # Process face and save
        img = img.resize((128,128), PIL.Image.BILINEAR)
        output = predict(img)
        fp_aft=tempfile.NamedTemporaryFile(suffix=str(i)+'aft'+filename,dir=save_dir.name,delete=False)
        output.save(fp_aft)

        # Add to list
        imgs = (os.path.relpath(fp_bef.name), os.path.relpath(fp_aft.name))
        img_paths.append(imgs)
    return img_paths

def predict_as_is(img_path, filename):
    # has issues with scaling
    img = vision.image.open_image(img_path)
    # get scale of image
    _, input_x, input_y = img.shape 
    img_scale = input_y/input_x
    # output will be of size output_y by output_x shape
    output_y = 512 * img_scale

    # prediction
    fp_bef = tempfile.NamedTemporaryFile('bef'+filename, dir = save_dir.name, delete=False)
    t1 = time.time()
    output = predict(img)
    print("predict-as-is time: ", time.time() - t1)
    img.save(fp_bef, 'PNG')
    fp_aft = tempfile.NamedTemporaryFile('aft'+filename, dir=save_dir.name, delete=False)
    output.save(fp_aft)
    
    img_paths = []
    img_paths.append(os.path.relpath(fp_bef.name))
    img_paths.append(os.path.relpath(fp_aft.name))
    return img_paths

'''
fn that receives post request, and downloads incoming file
High resolution file is generated from the model, and saved.
The filepath is sent back to client
'''
@app.route("/mobile_predict", methods=['POST'])

def predict():
    try:
        print(request.headers)
        data = request.files['file']
    except:
        keylist = [x for x in request.files]
        keylist_str = ','.join(keylist)
        wrong_key_str = "error: wrong key, \"file\" expected, got \"{}\" instead".format(keylist_str)
        print(wrong_key_str)
        return jsonify({"message":wrong_key_str})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        fp = tempfile.NamedTemporaryFile(suffix=filename, dir=save_dir.name, delete=False)
        file.save(fp.name)

        if request.files['do_landmark_detection'] == True:
            img_paths_list = generate_faces(fp.name, filename)
            img_paths_str = ','.join(img_paths_list)
            
        else :
            img_paths_list = predict_as_is(fp.name, filename)
            img_paths_str = ','.join(img_paths_list)
            

    # return filename with tokens to client
    return jsonify({'filenames':img_paths_str, "status":"ok"})

'''
function that sends file to client
the request from client will come with a (str)filename.
The image in the corresponding filename is returned to client
curl command:
curl http://6fd9b167.ngrok.io/download?filename="hi-res.png" -o from-server.png
'''
@app.route("/download", methods=['GET'])
def return_image():
    # form filepath from arguments provided by user
    filepath = 'Predicted/' + request.args.get('filename')
    print(filepath)
    try:
        if os.path.isfile(filepath):
            #check download then remove
            file_handle = open(filepath,'r')
            yield from file_handle
            file_handle.close()
            os.remove(filepath)
            # return send_file(filepath)

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
    app.run(host='0.0.0.0', port = 80)
