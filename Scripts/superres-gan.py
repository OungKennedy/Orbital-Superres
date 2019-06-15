'''
file to train super resolution model on SOC GPU.
pre- trains a generator and critic, then uses a GAN to further develop the two.
at the end of the file, the generator is exported.
dataset used: Pets dataset 
'''
#---------------
# pre-run checks: ensure fastai installed
# this python file is placed in Orbital/fastai/scripts
# if first run, ensure that the data is crappified
# if adding new data, remove old ones
#----------------
import fastai, os
from fastai.vision import *
from fastai.callbacks import *
from fastai.vision.gan import *
from PIL import Image

#working directory is Orbital/fastai
os.chdir("..")
base_dir = '/fastai'
dataset_url = 'https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet'
data_path = base_dir + "/data"
pet_path = untar_data(dataset_url, dest=data_path)
path_lr = pet_path/'lr'
path_hr = pet_path/'images'
bs, size = 32, 128
#bs, size = 24, 160
#bs, size = 8, 256
arch = models.resnet34
wd = 1e-3
y_range = (-3.,3.)
loss_gen = MSELossFlat()

class crappifier(object):
    #next iteration check gaussian salt and pepper noise
    def __init__(self, path_lr, path_hr):
        self.path_lr = path_lr
        self.path_hr = path_hr              
        
    def __call__(self, fn, i):       
        dest = self.path_lr/fn.relative_to(self.path_hr)    
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = PIL.Image.open(fn)
        targ_sz = resize_to(img, 96, use_min=True)
        img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
        w,h = img.size
        q = random.randint(10,70)
        img.save(dest, quality=q)

def prepare_data(base_dir):
    # check if data already exists
    il = ImageList.from_folder(path_hr)
    parallel(crappifier(path_lr,path_hr),il.items)

def get_data(bs, size, src, path_hr):
    data = (src.label_from_func(lambda x: path_hr/x.name)
    .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
    .databunch(bs=bs).normalize(imagenet_stats,do_y=True))
    data.c = 3
    return data

def create_gen_learner(data_gen, arch, wd, y_range, loss_func):
    return unet_learner(data_gen, arch, wd=wd, blur=True, norm_type=NormType.Weight,
    self_attention=True,y_range=y_range, loss_func=loss_func)

def save_preds(dl, model):
    i=0
    names = dl.dataset.items
    for b in dl:
        preds = model.pred_batch(batch=b,reconstruct=True)
        for o in preds:
            o.save(path_gen/names[i].name)
            i += 1

def get_crit_data(classes, src, bs, size, path):
    src = ImageList.from_folder(path, include=classes).split_by_rand_pct(0.1, seed=42)
    ll = src.label_from_folder(classes=classes)
    data = (ll.transform(get_transforms(max_zoom=2.), size=size)
           .databunch(bs=bs).normalize(imagenet_stats))
    data.c = 3
    return data

def create_critic_learner(data, metrics, loss_func):
    return Learner(data, gan_critic(), metrics=metrics, loss_func=loss_func,wd=wd)

if __name__ == "__main__":
    prepare_data(base_dir)

    # Pre-train generator
    src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1,seed=42)
    data_gen = get_data(bs,size,src, path_hr)
    learn_gen = create_gen_learner(data_gen,arch,wd,y_range,loss_gen)
    learn_gen.fit_one_cycle(1, pct_start=0.8)
    learn_gen.unfreeze()
    learn_gen.fit_one_cycle(1, slice(1e-6,1e-3))
    learn_gen.save('gen-pre2')

    #Save generated images
    name_gen = 'image_gen'
    path_gen = pet_path/name_gen
    path_gen.mkdir(exist_ok=True)
    save_preds(data_gen.fix_dl, learn_gen)

    #Train Critic
    learn_gen = None
    gc.collect()
    data_crit = get_crit_data(['image_gen','images'], src=src, bs=bs, size=size, path=pet_path)
    loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())
    learn_critic = create_critic_learner(data_crit,accuracy_thresh_expand,loss_critic)
    learn_critic.fit_one_cycle(1,1e-3)
    learn_critic.save('critic-pre2')

    #Combined pretrained model in GAN
    learn_crit = None
    learn_gen = None
    gc.collect()
    data_crit = get_crit_data(['lr','images'],src,bs=bs,size=size,path=pet_path)
    learn_crit = create_critic_learner(data_crit,None,loss_critic).load('critic-pre2')
    learn_gen = create_gen_learner(data_gen,arch,wd,y_range,loss_gen).load('gen-pre2')
    switcher = partial(AdaptiveGANSwitcher,critic_thresh=0.65)
    learn=GANLearner.from_learners(learn_gen,learn_crit,weights_gen=(1.,50.),show_img=False,switcher=switcher,
    opt_func=partial(optim.Adam,betas=(0.,0.99)), wd=wd)
    learn.callback_fns.append(partial(GANDiscriminativeLR,mult_lr=5.))
    lr = 1e-4
    learn.fit(1,lr)
    learn.save('gan-1c')
    learn.fit(1,lr/2)
    learn.save('gan-1c')

    #Export as pkl file
    learn_gen.export()
    


    
