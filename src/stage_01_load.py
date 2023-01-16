from src.utils.all_utils import read_yaml,create_directory
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm


def copy_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    n = len(list_of_files)
    for file in tqdm(list_of_files,total=n,desc="copying files"):
        src = os.path.join(source_download_dir,file)
        dest = os.path.join(local_data_dir,file)
        shutil.copy(src,dest)

def get_data(config_path):

    config = read_yaml(config_path)

    source_download_dirs = config['source_data_dir']
    
    local_dir = config['data_dirs']
    
    
    for souce_download, local_data_dir in tqdm(zip(source_download_dirs,local_dir),total=2, desc="list pf folders"):
        create_directory([local_data_dir])
        copy_file(souce_download,local_data_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    try:
        get_data(config_path=parsed_args.config)
    except Exception as e:
        raise e