import yaml
import os
import json
import shutil
import numpy as np
from tqdm import tqdm
import cv2
import imutils
from application_logging.logger import App_Logger

log_write = App_Logger()



def read_yaml(path_to_yaml: str,log_file):
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            log_write.log(log_file,f'yaml file: {path_to_yaml} loaded successfully')
        return content
    except Exception as e:
        log_write.log(log_file,f'error occure: {e}')
        raise e


def create_directory(dirs: list,log_file):
    try:
        for dir_path in dirs:
            os.makedirs(dir_path,exist_ok=True)
            log_write.log(log_file,f"directory is created at {dir_path}")
    except Exception as e:
        log_write.log(log_file,f"error occur in create_directory function: {e}")
        raise(e)

def get_data_local_sys(source_download_dir,local_data_dir,log_file):
    try:
        list_of_files = os.listdir(source_download_dir)
        n = len(list_of_files)
        for file in tqdm(list_of_files,total=n,desc="copying files"):
            src = os.path.join(source_download_dir,file)
            dest = os.path.join(local_data_dir,file)
            shutil.copy(src,dest)
        log_write.log(log_file,f"successfully get files from {source_download_dir} to {local_data_dir} local system")
    except Exception as e:
        log_write.log(log_file,f" error ocuur in get_data_local_sys function: {e}")
        raise e


def copy_test_file(source_download_dir,local_data_dir,log_file):
    try:
        list_of_files = os.listdir(source_download_dir)
        n = len(list_of_files)
        for(i,f) in enumerate(list_of_files):
            if i < 5:
                src = os.path.join(source_download_dir,f)
                dest = os.path.join(local_data_dir,f)
                shutil.copy(src,dest)
        
        log_write.log(log_file,f"successfully copy test files from {source_download_dir} to {local_data_dir} for testing model")
    except Exception as e:
        log_write.log(log_file,f" error ocuur in copy_test_file {e}")
        raise e

def copy_train_file(source_download_dir,local_data_dir,log_file):
    try:
        list_of_files = os.listdir(source_download_dir)
        n = len(list_of_files)
        for(i,f) in enumerate(list_of_files):
            if i < 0.8*n and i > 5:
                src = os.path.join(source_download_dir,f)
                dest = os.path.join(local_data_dir,f)
                shutil.copy(src,dest)
        log_write.log(log_file,f"successfully copy train files from {source_download_dir} to {local_data_dir} fot training model")
    except Exception as e:
        log_write.log(log_file,f" error occur in copy_train_file {e}")
        raise e



def copy_val_file(source_download_dir,local_data_dir,log_file):
    try:
        list_of_files = os.listdir(source_download_dir)
        n = len(list_of_files)
        for(i,f) in enumerate(list_of_files):
            if i > 0.8*n and i > 5:
                src = os.path.join(source_download_dir,f)
                dest = os.path.join(local_data_dir,f)
                shutil.copy(src,dest)
        log_write.log(log_file,f"successfully copy validations files from {source_download_dir} to {local_data_dir}")
    except Exception as e:
        log_write.log(log_file,f"error ocuur in copy_val_file function {e}")
        raise e

def load_data(dir_path,log_file, img_size=(100,100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    try:
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
        log_write.log(log_file,"successfully load data from resized and save as np.array")
        return X, y, labels
    except Exception as e:
        log_write.log(log_file,f" error occur in load_data function {e}")
        raise e



def crop_imgs(set_name,log_file,add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    try:
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
    
        log_write.log(log_file,f"successfully crop the image")
        return np.array(set_new)
    except Exception as e:
        log_write.log(log_file,f" error occur in crop_image function {e}")
        raise e

def save_new_images(x_set, y_set,path,log_file):
    i = 0
    try:
        for (img, imclass) in zip(x_set, y_set):
            if imclass == 0:
                cv2.imwrite(path +'no/'+str(i)+'.jpg', img)
            else:
                cv2.imwrite(path +'yes/'+str(i)+'.jpg', img)
            i += 1
        log_write.log(log_file,f" successfully save the crop the image in {path}")
    except Exception as e:
        log_write.log(log_file,f" error occur in save_new_image  function {e}")
        raise e





    
            

