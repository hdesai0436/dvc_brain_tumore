import tensorflow as tf
import joblib
import os
from src.utils.all_utils import get_timestamp
from application_logging.logger import App_Logger
log_write = App_Logger()


def  create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir,log_file):
    try:
        unique_name = get_timestamp("tb_logs")

        tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

        tb_callback_filepath = os.path.join(callbacks_dir, "tensorboard_cb.cb")
        joblib.dump(tensorboard_callback, tb_callback_filepath)
        log_write.log(log_file,f"tensorboard callback is being saved at {tb_callback_filepath}")
    except Exception as e:
        log_write.log(log_file,"error occur in create_and_save_tensorboard_callback: {e} ")
        raise e


def create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir,log_file):
    try:
        checkpoint_file_path = os.path.join(checkpoint_dir, "ckpt_model.h5")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_file_path,
            monitor = 'val_accuracy',
            save_best_only=True
        )

        ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
        joblib.dump(checkpoint_callback, ckpt_callback_filepath)
        log_write.log(log_file,f"tensorboard callback is being saved at {ckpt_callback_filepath}")
    except Exception as e:
        log_write.log(log_file,"error occur in create_and_save_checkpoint_callback: {e} ")
        raise e

def early_stopping(callbacks_dir,log_file):
    try:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
        mode='max',
        patience=6)
        er_callback_filepath = os.path.join(callbacks_dir, "early_stopping.cb")
        joblib.dump(early_stop, er_callback_filepath)
        log_write.log(log_file,"early_stopping callback created")
    except Exception as e:
        log_write.log(log_file,"error occur in early_stopping: {e} ")
        raise(e)





def get_callbacks(callback_dir_path,log_file):
    try:
        callback_path = [
            os.path.join(callback_dir_path, bin_file) for bin_file in os.listdir(callback_dir_path) if bin_file.endswith(".cb")
        ]

        callbacks = [
            joblib.load(path) for path in callback_path
        ]

        log_write.log(log_file,f"saved callbacks are loaded from {callback_dir_path}")

        return callbacks
    except Exception as e:
        log_write.log(f'error occur: {e}')
        raise e

