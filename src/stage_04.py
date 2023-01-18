from src.utils.all_utils import read_yaml,create_directory,load_data,crop_imgs,save_new_images
from src.utils.model import get_vgg_16_model,prepare_model
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
log_file = open("log_dir/stage04_log.txt", 'a+')


def prepare_base_model(config_path,params_path):
    config = read_yaml(config_path,log_file)
    params = read_yaml(params_path,log_file)

    
    artifacts_dir = config['artifacts']['artifacts_dir']

    base_model_dir = config['artifacts']['base_model_dir']
    base_model_name = config['artifacts']['base_model_name']

    base_model_dir_path = os.path.join(artifacts_dir,base_model_dir)
    create_directory([base_model_dir_path],log_file)

    base_model_path = os.path.join(base_model_dir_path,base_model_name)

    model = get_vgg_16_model(input_shape= params['IMAGE_SIZE'],model_path=base_model_path,log_file=log_file)

    model = prepare_model(
        model,
        classes = params['CLASSES'],
        freeze_all=True,
        freeze_till=None,
        learning_rate=params["LEARNING_RATE"],
        log_file=log_file
    )

    update_base_model_path = os.path.join(base_model_dir_path,config['artifacts']['update_base_model_name'])

    model.save(update_base_model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
        log_write.log(log_file,'>>>>>>>> stage 04 stated')
        prepare_base_model(config_path=parsed_args.config, params_path=parsed_args.params)
        log_write.log(log_file,'>>>>>>>> stage 04 end')
    except Exception as e:
        log_write.log(log_file,f'stage 04 error occur {e}')
        raise e

