import re
import random
import shutil
from PIL import Image, ImageEnhance
from skimage.transform import resize
from skimage.util import random_noise
from skimage.measure import label, regionprops
from scipy.ndimage.interpolation import rotate, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from os import listdir, walk, remove
import numpy as np
import nibabel as nib 
from random import uniform
from deepbrain import Extractor
import matplotlib.pyplot as plt



def unzip_nii(origin, destination):
    counter = 0 
    for root, dirs, files in walk(origin):
        for name in files:
            if name.endswith((".nii")):
                file_path = root + "/" + name
                counter += 1
                shutil.move(file_path, destination)

    return(str(counter) + " files have been moved to selected folder")

def Img_filter(path, des):
    files = listdir(path)
    counter = 0
    for name in files:
        fpath = path + "/" + name
        nii_img = nib.load(fpath)
        try:
            data = nii_img.get_data()
            img = np.asarray(data)
            s = img.shape
            if s[0] < s[2]:
                name = name.replace(".nii", "")
                newpath = des + "/" + name
                np.save(newpath,data)
            else: 
                counter += 1

        except: 
            counter += 1
    print(str(counter) + " images not meeting the selection criteria/failed to load")

# resize the images to (64,64,64)
def resize_64_64_64(image, ideal_shape = (64, 64, 64)):
    # go along the x axis,resize images on the y and z axis
    img_new1 = np.zeros((image.shape[0],ideal_shape[1], ideal_shape[2]))
    for i in range(image.shape[0]):
        img = image[i,:,:]
        img_new1[i,:,:] = resize(img, (ideal_shape[1], ideal_shape[2]), anti_aliasing=True)
    
    # go along the y axis, resize images on the x and z axis
    img_new2 = np.zeros(ideal_shape)
    for i in range(img_new1.shape[1]):
        img = img_new1[:,i,:]
        img_new2[:,i,:] = resize(img, (ideal_shape[0], ideal_shape[2]), anti_aliasing=True)
    return img_new2

# intensity-nomalization
def intensilty_normalization(image): 
  img_f = image.flatten()
  i_range = img_f[np.argmax(img_f)] - img_f[np.argmin(img_f)]
  image = image - img_f[np.argmin(img_f)]
  img_normalized = np.float32(image/i_range)
  return img_normalized

# create a preprocessing procedure
def final_preprocessing(img):
    img_skull = skull_stripper(img)
    img_brain_cropping = brain_cropping(img_skull)
    img_std = resize_64_64_64(img_brain_cropping)
    img_nor = intensilty_normalization(img_std)
    return img_nor

# assign labels to the images in a folder
def label_assign(path, ref):
    files = listdir(path)
    label_assign = []

    for names in files:
        sub_id = "".join(names)
        if sub_id.endswith(("_pat_processed.npy")):
            new_lab = 1
        else:
            new_lab = 0
        label_assign.append(new_lab)
    print(str(len(label_assign)) + " labels have been assigned to the data")
    return label_assign


def brain_cropping(image):
    img_bo = image > 0
    img_labeled = label(img_bo)
    bounding_box = regionprops(img_labeled)
    bb = bounding_box[0].bbox
    img_brain_cropping = image[bb[0]:bb[3],
               bb[1]:bb[4],
               bb[2]:bb[5]]
    return img_brain_cropping

def skull_stripper(image):
    ext = Extractor()
    prob = ext.run(image)
    mask = prob < 0.7
    img_filtered = np.ma.masked_array(image, mask = mask)
    img_filtered = img_filtered.filled(0)
    return img_filtered

