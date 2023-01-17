from src.utils.all_utils import read_yaml,create_directory,load_data,crop_imgs,save_new_images
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import imutils


def preprocess(config_path):
    IMG_SIZE = (224,224)
    config = read_yaml(config_path)
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
    
    
        
    x_test,y_test,_ = load_data(test_data_path,IMG_SIZE)
    x_train,y_train,labels= load_data(train_data_path)
    x_val,y_val,_= load_data(val_data_path)

    x_train_crop = crop_imgs(set_name=x_train)
    x_val_crop = crop_imgs(set_name=x_val)
    x_test_crop = crop_imgs(set_name=x_test)

    
    for i in test_crop_path:
        create_directory([i])
    for i in train_crop_path:
        create_directory([i])
    for i in val_crop_path:
        create_directory([i])

    
        
    
    save_new_images(x_test_crop,y_test,test_crop_dir_path)
    save_new_images(x_train_crop,y_train,train_crop_dir_path)
    save_new_images(x_val_crop,y_val,val_crop_dir_path)

       

    

    
    


    




    
    



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    try:
        preprocess(config_path=parsed_args.config)
    except Exception as e:
        raise e