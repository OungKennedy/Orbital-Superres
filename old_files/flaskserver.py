from flask import Flask, jsonify, request, send_file
import os
import time
import fastai
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
import torchvision
import Scripts.utils
from fastai import (basic_train, vision)
from Scripts.size_change import change_output_size
app = Flask(__name__)

#load model 
learn_noise = None
learn = None

#init feature loss class
class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',] + [f'feat_{i}' for i in range(len(layer_ids))
              ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input,target)]
        self.feat_losses += [base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

#function that loads model named export.pkl
#export.pkl is in same directory as this python file
def load_model():
    global learn
    global learn_noise
    learn = fastai.basic_train.load_learner('models','faces_featloss_2206.pkl')
    learn_noise = fastai.basic_train.load_learner('models','noise_featloss_2606.pkl')
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
    # save uploaded file to Uploads folder
    scale = 2
    filename = 'Uploads/uploaded_img-{}.png'.format(int(time.time()))
    data.save(filename)
    # open images as fastai.Vision.image.image type
    image = vision.image.open_image(filename)
    channel,input_x,input_y = image.shape
    #boolean for inference by patching
    do_ibp = True
    if input_y < 500 or input_x < 500:
        do_ibp = False

    #denoise image before super resolution
    change_output_size(learn_noise,input_x,input_y)
    img_denoised = learn_noise.predict(image)[0]

    if not do_ibp:
        change_output_size(learn, input_x*scale,input_y*scale)
        img_hr = learn.predict(img_denoised)[0]
    else:
        '''
        get shape of output (deciding the size of final, super resolution image)
        limit of 1000 due to long processing time for large images
        further implementation might feature patching technique
        which cuts the image into patches and processes them individually
        before stitching them back together
        ''' 
        #NOTE THIS LINE WAS CHANGED
        patch_size = 500 
        change_output_size(learn,patch_size*input_x,patch_size*input_y)
        img_hr = Scripts.utils.predict(learn,img_denoised)
        
    # clamp all pixel values in image to be in range of 0 to 1
    # save file with unique filename in 'Predicted' folder
    hr_filename = 'hi-res-{}.png'.format(int(time.time()))
    hr_savepath = 'Predicted/' + hr_filename
    #NOTE save function changed, image will become negative when using ibp
    img_hr.save(hr_savepath)
    # add token to simplify tokenisation on client side
    hr_filename = '%' + hr_filename + '%'
    # return filename with tokens to client
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
