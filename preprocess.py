#### Required imports ####


import keras
import tensorflow as tf
from glob import glob
import random, os, datetime

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model



### Extracting Photos ###


import os
import random
import zipfile
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, metrics
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models  import Sequential, load_model
from tensorflow.keras.metrics import Accuracy, AUC
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def extract_zip(zip_path, extract_to):
    """Extract the ZIP file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
def get_image_paths(root_dir, num_images=None):
    """Get the paths of images in the specified directory."""
    all_images = []
    for extension in ['*.jpg', '*.jpeg', '*.png']:
        all_images.extend(glob(os.path.join(root_dir, '**', extension), recursive=True))
    if num_images is None:
        return all_images
    else:
        return random.sample(all_images, min(num_images, len(all_images)))
def display_images_with_labels(img_list, root_dir):
    """Display images in a grid format with their real class labels."""
    plt.figure(figsize=(15, 6))
    for i, img_path in enumerate(img_list):
        img = image.load_img(img_path)
        img = image.img_to_array(img, dtype=np.uint8)
        label = img_path.split('/')[-2]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.squeeze())
        plt.axis('off')
        plt.title(f'{label}')
    plt.tight_layout()
    plt.show()
zip_file_path = '/content/archive (5).zip'  
extract_to_dir = '/content/yeni deneme'  
extract_zip(zip_file_path,extract_to_dir)
reals = "/content/yeni deneme/Garbage classification/Garbage classification"
root_dir = reals
num_images_to_display = 10  
image_paths = get_image_paths(root_dir, num_images=num_images_to_display)  
display_images_with_labels(image_paths, root_dir)

