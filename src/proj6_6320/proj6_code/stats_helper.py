import glob
import os
import numpy as np

from PIL import Image
from sklearn.preprocessing import StandardScaler
from image_loader import ImageLoader

def compute_mean_and_std(dir_name: str) -> (np.array, np.array):
  '''
  Compute the mean and the standard deviation of the dataset.

  Note: convert the image in grayscale and then in [0,1] before computing mean
  and standard deviation

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  mean = None
  std = None

  ############################################################################
  # Student code begin
  ############################################################################
  root_dir = dir_name
  match_file = root_dir + '/*/*/*.jpg'
  image_dirs = glob.glob(match_file); num_imgs = len(image_dirs)
  
  # Open images, convert to grayscale and normalize to [0 1]
  data = [np.array(Image.open(image_dir).convert(mode='L')).flatten() / 255 for image_dir in image_dirs]
  
  data = np.concatenate(data, axis=0)
  
  mean = np.array([data.mean()])
  std = np.array([np.sqrt(data.var())])
  ############################################################################
  # Student code end
  ############################################################################
  return mean, std
