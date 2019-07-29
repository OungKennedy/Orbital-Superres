'''
function to change output size of model
'''
import fastai, os
from fastai.vision import *
from fastai import basic_train

def change_output_size(model, output_x, output_y):
    path = Path(__file__).parent
    databunch = (vision.data.ImageImageList.from_folder(path, ignore_empty=True).split_none()
          .label_from_func(lambda x: x)
          .transform(vision.transform.get_transforms(), size=(output_x, output_y), tfm_y=True)
          .databunch(bs=2, no_check=True).normalize(vision.data.imagenet_stats, do_y=True))
    databunch.c = 3
    model.data = databunch
    

