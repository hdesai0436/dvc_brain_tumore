from src.utils.all_utils import read_yaml,create_directory,copy_test_file,copy_train_file,copy_val_file
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
from application_logging.logger import App_Logger
log_write = App_Logger()
log_file = open("log_dir/stage02_log.txt", 'a+')

def split_data(config_path):
    config = read_yaml(config_path,log_file)
    local_dir_path = config['data_dirs']
    split_data_path = config['test_dir']
    train_data_path = config['train_dirs']
    val_data_path = config['val_dirs']
   
    for souce_download, local_data_dir in tqdm(zip(local_dir_path,split_data_path),total=2, desc="list pf folders"):
        create_directory([local_data_dir],log_file)
        copy_test_file(souce_download,local_data_dir,log_file)
    
    
    for souce_download, train_data in tqdm(zip(local_dir_path,train_data_path),total=2, desc="list pf folders"):
        create_directory([train_data],log_file)
        copy_train_file(souce_download,train_data,log_file)
    

    for souce_download, val_data in tqdm(zip(local_dir_path,val_data_path),total=2, desc="list pf folders"):
        create_directory([val_data],log_file)
        copy_val_file(souce_download,val_data,log_file)
    
        

    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    try:
        log_write.log(log_file,'>>>>>>>> stage 02 stated')
        split_data(config_path=parsed_args.config)
        log_write.log(log_file,'>>>>>>>> stage 02 end')
    except Exception as e:
        raise e



