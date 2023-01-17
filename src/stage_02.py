from src.utils.all_utils import read_yaml,create_directory,copy_test_file,copy_train_file,copy_val_file
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm

def split_data(config_path):
    config = read_yaml(config_path)
    local_dir_path = config['data_dirs']
    split_data_path = config['test_dir']
    train_data_path = config['train_dirs']
    val_data_path = config['val_dirs']
   
    for souce_download, local_data_dir in tqdm(zip(local_dir_path,split_data_path),total=2, desc="list pf folders"):
        create_directory([local_data_dir])
        copy_test_file(souce_download,local_data_dir)
    
    for souce_download, train_data in tqdm(zip(local_dir_path,train_data_path),total=2, desc="list pf folders"):
        create_directory([train_data])
        copy_train_file(souce_download,train_data)

    for souce_download, val_data in tqdm(zip(local_dir_path,val_data_path),total=2, desc="list pf folders"):
        create_directory([val_data])
        copy_val_file(souce_download,val_data)

    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    try:
        split_data(config_path=parsed_args.config)
    except Exception as e:
        raise e



