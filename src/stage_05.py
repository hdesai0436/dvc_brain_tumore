from src.utils.all_utils import read_yaml,create_directory,load_data,crop_imgs,save_new_images
from src.utils.callback import create_and_save_checkpoint_callback,create_and_save_tensorboard_callback,early_stopping
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
log_file = open("log_dir/stage05_log.txt", 'a+')


def prepare_callback(config_path,params_path):
    config = read_yaml(config_path,log_file)
    params = read_yaml(params_path,log_file)

    artifacts_dir = config['artifacts']['artifacts_dir']
    tensorboard_log_dir = os.path.join(artifacts_dir, 
    config['artifacts']["TENSORBOARD_ROOT_LOG_DIR"])

    checkpoint_dir = os.path.join(artifacts_dir,  config['artifacts']["checkpoint_dir"])
    
    callbacks_dir = os.path.join(artifacts_dir,  config['artifacts']["callback_dir"])

    

    create_directory([
        tensorboard_log_dir,
        checkpoint_dir,
        callbacks_dir
    ],log_file=log_file)

    create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir,log_file=log_file)
    create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir,log_file=log_file)
    early_stopping(callbacks_dir,log_file=log_file)
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
        log_write.log(log_file,'>>>>>>>> stage 05 stated')
        prepare_callback(config_path=parsed_args.config, params_path=parsed_args.params)
        log_write.log(log_file,'>>>>>>>> stage 05 end')
    except Exception as e:
        log_write.log(log_file,f'stage 04 error occur {e}')
        raise e