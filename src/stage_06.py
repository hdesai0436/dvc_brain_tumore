from src.utils.all_utils import read_yaml,create_directory,load_data,crop_imgs,save_new_images
from src.utils.callback import get_callbacks, create_and_save_checkpoint_callback,create_and_save_tensorboard_callback,early_stopping
from src.utils.model import load_full_model,get_unique_path_to_save_model
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import scipy
import tensorflow as tf

from application_logging.logger import App_Logger
log_write = App_Logger()
log_file = open("log_dir/stage06_log.txt", 'a+')

def train_model(config_path,params_path):
    config = read_yaml(config_path,log_file)
    params = read_yaml(params_path,log_file)

    artifacts_dir = config['artifacts']['artifacts_dir']
    test_data_path = config['artifacts']['test_crop_dir']
    train_data_path = config['artifacts']['train_crop_dir']
    val_data_path = config['artifacts']['val_crop_dir']


    train_model_dir_path = os.path.join(artifacts_dir,config['artifacts']['train_model_dir'])

    create_directory([train_model_dir_path],log_file=log_file)

    untrain_full_model_path = os.path.join(artifacts_dir,config['artifacts']['base_model_dir'],config['artifacts']['update_base_model_name'])

    model = load_full_model(untrain_full_model_path,log_file=log_file)

    callback_dir_path = os.path.join(artifacts_dir,config['artifacts']['callback_dir'])

    callback = get_callbacks(callback_dir_path,log_file=log_file)

    train_generator = tf.keras.preprocessing.image.ImageDataGenerator( 
                                    rotation_range=15,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range=0.1,
                                    brightness_range=[0.5, 1.5],
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    validation_split=0.15,
                                    preprocessing_function = tf.keras.applications.vgg16.preprocess_input ) # VGG16 preprocessing

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input ) 

    traingen = train_generator.flow_from_directory(train_data_path,target_size=(224,224),subset='training',)
    valgen = train_generator.flow_from_directory(val_data_path,target_size=(224,224),subset='validation')
    testgen = test_generator.flow_from_directory(test_data_path,target_size=(224,224))

    log_write.log(log_file,'trainimg started')

    model.fit(
    traingen,
    steps_per_epoch=50,
    epochs=30,
    validation_data=valgen,
    validation_steps=25,
    callbacks=callback
    )

    train_model_dir_path = os.path.join(artifacts_dir,config['artifacts']['train_model_dir'])
    create_directory([train_model_dir_path],log_file=log_file)
    model_file_path = get_unique_path_to_save_model(train_model_dir_path)
    model.save(model_file_path)
    log_write.log(log_file,"trainning complated")





if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
        log_write.log(log_file,'>>>>>>>> stage 06 stated')
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        log_write.log(log_file,'>>>>>>>> stage 06 end')
    except Exception as e:
        log_write.log(log_file,f'stage 06 error occur {e}')
        raise e


