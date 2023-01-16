import yaml
import os
import json
import shutil

def read_yaml(path_to_yaml: str):
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content


def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path,exist_ok=True)


def copy_test_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    n = len(list_of_files)
    for(i,f) in enumerate(list_of_files):
        if i < 5:
            src = os.path.join(source_download_dir,f)
            dest = os.path.join(local_data_dir,f)
            shutil.copy(src,dest)


def copy_train_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    n = len(list_of_files)
    for(i,f) in enumerate(list_of_files):
        if i < 0.8*n and i > 5:
            src = os.path.join(source_download_dir,f)
            dest = os.path.join(local_data_dir,f)
            shutil.copy(src,dest)


def copy_val_file(source_download_dir,local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    n = len(list_of_files)
    for(i,f) in enumerate(list_of_files):
        if i > 0.8*n and i > 5:
            src = os.path.join(source_download_dir,f)
            dest = os.path.join(local_data_dir,f)
            shutil.copy(src,dest)



    
            

