import tensorflow as tf
import os
import io
from application_logging.logger import App_Logger
log_write = App_Logger()
from src.utils.all_utils import get_timestamp

def _get_model_summary(model):
    with io.StringIO() as stream:
        model.summary(
            print_fn=lambda x: stream.write(f"{x}\n")
        )
        summary_str = stream.getvalue()
    return summary_str


def get_vgg_16_model(input_shape,model_path,log_file):
    try:
        model = tf.keras.applications.vgg16.VGG16(
            input_shape= input_shape,
            weights='imagenet',
            include_top = False
        )

        model.save(model_path)
        log_write.log(log_file,f"VGG16 base model saved at: {model_path}")
        return model
    except Exception as e:
        log_write.log(log_file,f"Error occur in get_vgg_16_model function {e}")
        raise e


def prepare_model(base_model,classes,freeze_all,freeze_till,learning_rate,log_file):
    try:
        if freeze_all:
            for layer in base_model.layers:
                layer.trainable=False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in base_model.layers[:-freeze_till]:
                layer.trainable = False

        ## add our layers to the base model
        flatten_in = tf.keras.layers.Flatten()(base_model.output)

        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs = prediction
        )

        full_model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss = tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
        log_write.log(log_file,f"model summary {_get_model_summary(full_model)}")
        log_write.log(log_file,"custom model is compiled and ready to be trained")

        return full_model
    except Exception as e:
        log_write.log(log_file,"error occur in prepare_model function: {e}")
        raise e


def load_full_model(model_path,log_file):
    try:
        model = tf.keras.models.load_model(model_path)
        log_write.log(log_file,f"untrained model is read from: {model_path}")
        return model
    except Exception as e:
        log_write.log(log_file,f"Error occur in load_full_model")
        raise e

def get_unique_path_to_save_model(trained_model_dir, model_name="model"):
    timestamp = get_timestamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    return unique_model_path
        
        



    