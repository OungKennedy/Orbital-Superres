import numpy as np
import fastai, os
from fastai.vision import *
from fastai import basic_train
##################################################
def predict(model, input_image, by_patch_of_size=500, batch_size=4, padding_size=2, scale=2):    
  input_image_array = image2np(input_image.data)
  if by_patch_of_size:
    patches, p_shape = split_image_into_overlapping_patches(
      input_image_array, patch_size=by_patch_of_size, padding_size=padding_size
    )
    #transpose arrays since np is in H x W x C but torch image is in C x H x W
    batch_transposed = [arr.transpose(2,0,1) for arr in patches]
    #convert to fastai image format
    batch_fa_image = [fastai.vision.Image(torch.from_numpy(arr)) for arr in batch_transposed]
    # return patches
    for i in range(0, len(patches), batch_size):
      batch = [(learn.predict(image)[1]).numpy().transpose((1, 2, 0)) for image in batch_fa_image[i:i+batch_size]]
      #obtain a numpy array of all the predictions, appended
      if i == 0:
        collect = batch
      else:
        collect = np.append(collect, batch, axis=0)
    #scale is scale of super resolution?
    padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
    scaled_image_shape = tuple(np.multiply(input_image_array.shape[0:2], scale)) + (3,)
    sr_img = stitch_together(
      collect,
      padded_image_shape=padded_size_scaled,
      target_shape=scaled_image_shape,
      padding_size=padding_size * scale,
    )

  else:
    lr_img = process_array(input_image_array)
    sr_img = model.predict(lr_img)[1]

  sr_img = process_output(sr_img)
  final_img = fastai.vision.Image(torch.from_numpy(sr_img.transpose(2,0,1)))
  return final_img

import numpy as np
#consider image2np

def process_array(image_array, expand=True):
  '''process a 3d array into a scaled, 4d batch of size 1'''
#   image_batch = image_array/255.0
  if expand:
    image_batch = np.expand(image_array, axis=0)
    return image_batch
  else:
    return image_array

def split_image_into_overlapping_patches(image_array, patch_size, padding_size=2):
    """ Splits the image into partially overlapping patches.
    The patches overlap by padding_size pixels.
    Pads the image twice:
        - first to have a size multiple of the patch size,
        - then to have equal padding at the borders.
    Args:
        image_array: numpy array of the input image.
        patch_size: size of the patches from the original image (without padding).
        padding_size: size of the overlapping area.
    """

    xmax, ymax,_ = image_array.shape
    x_remainder = xmax % patch_size
    y_remainder = ymax % patch_size

    # modulo here is to avoid extending of patch_size instead of 0
    x_extend = (patch_size - x_remainder) % patch_size
    y_extend = (patch_size - y_remainder) % patch_size

    # make sure the image is divisible into regular patches
    extended_image = np.pad(image_array, ((0, x_extend), (0, y_extend), (0, 0)), 'edge')

    # add padding around the image to simplify computations
    padded_image = pad_patch(extended_image, padding_size, channel_last=True)

    xmax, ymax, _ = padded_image.shape
    patches = []

    x_lefts = range(padding_size, xmax - padding_size, patch_size)
    y_tops = range(padding_size, ymax - padding_size, patch_size)

    for x in x_lefts:
        for y in y_tops:
            x_left = x - padding_size
            y_top = y - padding_size
            x_right = x + patch_size + padding_size
            y_bottom = y + patch_size + padding_size
            patch = padded_image[x_left:x_right, y_top:y_bottom, :]
            patches.append(patch)

    return np.array(patches), padded_image.shape

def unpad_patches(image_patches, padding_size):
  return image_patches[:, padding_size:-padding_size, padding_size:-padding_size, :]

def process_output(output_tensor):
  '''Transforms 4d output tensor into a suitable image format'''
  sr_img = output_tensor.clip(0,1) * 255
  sr_img = np.uint8(sr_img)
  return sr_img

def pad_patch(image_patch, padding_size, channel_last=True):
  """ Pads image_patch with with padding_size edge values. """

  if channel_last:
    return np.pad(
      image_patch,
      ((padding_size, padding_size), (padding_size, padding_size), (0, 0)),
      'edge',
    )
  else:
    return np.pad(
      image_patch,
      ((0, 0), (padding_size, padding_size), (padding_size, padding_size)),
      'edge',
    )

def stitch_together(patches, padded_image_shape, target_shape, padding_size=4):
    """ Reconstruct the image from overlapping patches.
    After scaling, shapes and padding should be scaled too.
    Args:
        patches: patches obtained with split_image_into_overlapping_patches
        padded_image_shape: shape of the padded image contructed in split_image_into_overlapping_patches
        target_shape: shape of the final image
        padding_size: size of the overlapping area.
    """

    xmax, ymax, _ = padded_image_shape
    patches = unpad_patches(patches, padding_size)
    patch_size = patches.shape[1]
    n_patches_per_row = ymax // patch_size

    complete_image = np.zeros((xmax, ymax, 3))

    row = -1
    col = 0
    for i in range(len(patches)):
        if i % n_patches_per_row == 0:
            row += 1
            col = 0
        complete_image[
            row * patch_size : (row + 1) * patch_size, col * patch_size : (col + 1) * patch_size, :
        ] = patches[i]
        col += 1
    return complete_image[0 : target_shape[0], 0 : target_shape[1], :]