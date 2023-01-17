import yaml
import os
import json
import shutil
import numpy as np
from tqdm import tqdm
import cv2
import imutils

def read_yaml(path_to_yaml: str):
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content


def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)


def copy_test_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    n = len(list_of_files)
    for(i,f) in enumerate(list_of_files):
        if i < 5:
            src = os.path.join(source_download_dir,f)
            dest = os.path.join(local_data_dir,f)
            shutil.copy(src,dest)


def copy_train_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    n = len(list_of_files)
    for(i,f) in enumerate(list_of_files):
        if i < 0.8*n and i > 5:
            src = os.path.join(source_download_dir,f)
            dest = os.path.join(local_data_dir,f)
            shutil.copy(src,dest)


def copy_val_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    n = len(list_of_files)
    for(i,f) in enumerate(list_of_files):
        if i > 0.8*n and i > 5:
            src = os.path.join(source_download_dir,f)
            dest = os.path.join(local_data_dir,f)
            shutil.copy(src,dest)


def load_data(dir_path, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    resize = cv2.resize(img,(224,224))
                    X.append(resize)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels

def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        resize_img = cv2.resize(new_img,(240,240))
        set_new.append(resize_img)
   

    return np.array(set_new)

def save_new_images(x_set, y_set,path):
    i = 0
   
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(path +'no/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(path +'yes/'+str(i)+'.jpg', img)
        i += 1





    
            

