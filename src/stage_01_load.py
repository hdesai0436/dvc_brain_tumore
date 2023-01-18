from src.utils.all_utils import read_yaml,create_directory,get_data_local_sys
import argparse
import pandas as pd
import os
import shutil
from tqdm import tqdm
from application_logging.logger import App_Logger

log_write = App_Logger()
log_file = open("log_dir/stage01_log.txt", 'a+')




def get_data(config_path):

    config = read_yaml(config_path,log_file)

    source_download_dirs = config['source_data_dir']
    
    local_dir = config['data_dirs']
    
    
    for souce_download, local_data_dir in tqdm(zip(source_download_dirs,local_dir),total=2, desc="list pf folders"):
        create_directory([local_data_dir],log_file)
        get_data_local_sys(souce_download,local_data_dir,log_file)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    parsed_args = args.parse_args()
    try:
        log_write.log(log_file,'>>>>>>>> stage 01 stated')
        get_data(config_path=parsed_args.config)
        log_write.log(log_file,'>>>>>>>> stage 01 ended')
    except Exception as e:
        log_write.log(log_file,f'stage 01 error occur {e}')
        raise e