import os
import time
from io import BytesIO

from fastai.basic_train import load_learner, np
from fastai.vision import Image, PIL, pil2tensor, image2np, open_image
from flask import Flask, jsonify, request, send_file, render_template,g
import tempfile 

from werkzeug.utils import secure_filename

from face_detection import LandmarksDetector, image_align
from Scripts.helpers.feature_loss import FeatureLoss
from Scripts.size_change import change_output_size

from concurrent.futures import ThreadPoolExecutor
import __main__
__main__.FeatureLoss = FeatureLoss 

app = Flask(__name__)
#why is this required ?
app.config['SECRET_KEY'] = 'ysecretkeys'
ALLOWED_EXTENSIONS = {'png','jpg','jpeg'}
save_dir = tempfile.TemporaryDirectory(dir="static")


#load model from models directory
model = load_learner('models','resnet34-enhance.pkl')
landmarks_detector = LandmarksDetector()
print('model loaded')

@app.route('/_hello_world')
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def predict(img):
    x = pil2tensor(img, np.float32)
    x.div_(255)
    output = model.predict(Image(x))[0]
    return output

def generate_faces(img_path, filename):
    print("generating faces\n")
    img_paths = []
    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(img_path),start=1):
        # Extract face and save at 512px
        img = image_align(img_path, face_landmarks)
        fp_bef = tempfile.NamedTemporaryFile(suffix=str(i)+'bef'+filename, dir=save_dir.name, delete=False)
        img.save(fp_bef,'PNG')

        # Process face and save
        img = img.resize((128,128), PIL.Image.BILINEAR)
        output = predict(img)
        fp_aft=tempfile.NamedTemporaryFile(suffix=str(i)+'aft'+filename,dir=save_dir.name,delete=False)
        output.save(fp_aft)

        # Add to list
        imgs = (os.path.relpath(fp_bef.name), os.path.relpath(fp_aft.name))
        img_paths.extend(imgs)
    return img_paths

def predict_as_is(img_path, filename):
    print("predicting as is\n")
    # has issues with scaling
    img = PIL.Image.open(img_path) 
    # get scale of image
    input_x, input_y = img.size
    img_scale = input_y/input_x
    # output will be of size output_y by output_x shape
    output_y = round(512 * img_scale)
    if output_y % 2 == 1:
        output_y -= 1
    img = img.resize((128,round(128 * img_scale)), PIL.Image.BILINEAR)
    # prediction
    fp_bef = tempfile.NamedTemporaryFile(suffix='bef'+filename, dir = save_dir.name, delete=False)
    t1 = time.time()
    change_output_size(model,output_y,512)
    output = predict(img)
    print(output.shape)
    change_output_size(model, 512,512)
    print("predict-as-is time: ", time.time() - t1)
    img.save(fp_bef, 'PNG')
    fp_aft = tempfile.NamedTemporaryFile(suffix='aft'+filename, dir=save_dir.name, delete=False)
    output.save(fp_aft)
    
    # Add to list
    img_paths = [os.path.relpath(fp_aft.name)]
    return img_paths

'''
fn that receives post request, and downloads incoming file
High resolution file is generated from the model, and saved.
The filepath is sent back to client
'''
@app.route("/mobile_predict", methods=['POST'])

def handle_incoming_post():
    try:
        print(request.headers)
        file = request.files['file']
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
        print("file saved as " + str(fp.name) + '\n')
        # temporary dummy for boolean to do landmark detection
        do_landmark_detection = request.form.get("landmark_detection") == "true"
        if  do_landmark_detection:
            print("doing landmark detection\n")
            img_paths_list = generate_faces(fp.name, filename)
            
        else :
            print("do_landmark_detection != True\n")
            img_paths_list = predict_as_is(fp.name, filename)
        
        #prepend url to complete download link
        img_urls_list = ["http://35.197.17.49/download?filename="+str(i) for i in img_paths_list]
        img_urls_str = ','.join(img_urls_list)
        img_urls_str = '%' + img_urls_str + '%'
        print(img_urls_str + '\n')
    # return filename with tokens to client
    return jsonify({'filenames':img_urls_str, "status":"ok"})

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
    filepath = request.args.get('filename')
    print("filepath is " + filepath + '\n')
    try:
        if os.path.isfile(filepath):
            #check download then remove
            return send_file(filepath)

        else:
            print("file not found")
            return jsonify({'message':'file not found'})
    except Exception as e:
        return jsonify({'message':'error encountered','error message':str(e)})

@app.route('/', methods=['POST','GET'])
def index():
    print("entered function")
    g.files = []
    print("1")
    if request.method == "POST":
        print("2")
        file =request.files["file"]
        print("3")
        if file and allowed_file(file.filename):
            print("4")
            filename = secure_filename(file.filename)

            fp = tempfile.NamedTemporaryFile(suffix=filename, dir=save_dir.name, delete=False)
            file.save(fp.name)
            print("before gen face")
            img_paths = generate_faces(fp.name, filename)
            print("aft gen face")
            return render_template("index.html",original_fp=os.path.relpath(fp.name),fps=[tuple(img_paths)])
    return render_template("index.html", original_fp="static/img.jpg",fps=[("static/img_bef.png","static/img_aft.png")])
