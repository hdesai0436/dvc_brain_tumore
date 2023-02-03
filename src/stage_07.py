import argparse
from src.utils.all_utils import read_yaml
import tensorflow as tf
import os
import numpy as np





def test_model(config_path,params_path):
    config = read_yaml(config_path)
    test_data_path = config['artifacts']['test_crop_dir']
    artifacts_dir = config['artifacts']['artifacts_dir']
    checkpoint_dir = os.path.join(artifacts_dir,  config['artifacts']["checkpoint_dir"])
    checkpoint_file_path = os.path.join(checkpoint_dir, "ckpt_model.h5")

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input )
    testgen = test_generator.flow_from_directory(test_data_path,target_size=(224,224))

    true_classes = testgen.classes
    model = tf.keras.models.load_model(checkpoint_file_path)

    pred = model.predict(testgen)
    pred1 = np.argmax(pred)
    loss,acc = model.evaluate(true_classes,pred1)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config","-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()
    try:
       
        test_model(config_path=parsed_args.config, params_path=parsed_args.params)
        
    except Exception as e:
       
        raise e