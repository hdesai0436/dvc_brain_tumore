from src.utils.all_utils import read_yaml,create_directory,load_data,crop_imgs,save_new_images
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import imutils

from application_logging.logger import App_Logger
log_write = App_Logger()
log_file = open("log_dir/stage03_log.txt", 'a+')


def preprocess(config_path):
    IMG_SIZE = (224,224)
    config = read_yaml(config_path,log_file)
    artifacts = config["artifacts"]
    test_data_path = config['artifacts']['test_data_dir']
    train_data_path = config['artifacts']['train_data_dir']
    val_data_path = config['artifacts']['val_data_dir']

    test_crop_path = config['test_dir_crop']
    test_crop_dir_path = config['artifacts']['test_crop_dir']

    train_crop_path = config['train_dir_crop']
    train_crop_dir_path = config['artifacts']['train_crop_dir']

    val_crop_path = config['val_dir_crop']
    val_crop_dir_path = config['artifacts']['val_crop_dir']
    
    
        
    x_test,y_test,_ = load_data(test_data_path,log_file,IMG_SIZE)
    
    x_train,y_train,labels= load_data(train_data_path,log_file,IMG_SIZE)
    x_val,y_val,_= load_data(val_data_path,log_file,IMG_SIZE)

    x_train_crop = crop_imgs(set_name= x_train,log_file=log_file)
    x_val_crop = crop_imgs(set_name=x_val, log_file=log_file)
    x_test_crop = crop_imgs(set_name=x_test, log_file=log_file)

    
    for i in test_crop_path:
        create_directory([i],log_file)
    for i in train_crop_path:
        create_directory([i],log_file)
    for i in val_crop_path:
        create_directory([i],log_file)

    
        
    
    save_new_images(x_test_crop,y_test,test_crop_dir_path,log_file)
    save_new_images(x_train_crop,y_train,train_crop_dir_path,log_file)
    save_new_images(x_val_crop,y_val,val_crop_dir_path,log_file)

       

    

    
    


    




    
    



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    try:
        preprocess(config_path=parsed_args.config)
    except Exception as e:
        raise e