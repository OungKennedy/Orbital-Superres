'''
function to change output size of model
'''
import fastai, os
from fastai.vision import *
from fastai import basic_train

def change_output_size(learn, output_x, output_y):
    databunch = (vision.data.ImageImageList.from_folder('Predicted').split_none()
          .label_from_func(lambda x: x)
          .transform(vision.transform.get_transforms(), size=(output_x, output_y), tfm_y=True)
          .databunch(bs=2, no_check=True).normalize(vision.data.imagenet_stats, do_y=True))
    databunch.c = 3
    learn.data = databunch
    

