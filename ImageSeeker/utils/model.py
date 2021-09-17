from ImageSeeker.utils.config import configureData
from ImageSeeker.utils.config import configureModel
import tensorflow as tf
import os

config_data = configureData()
config_model = configureModel()

def get_model():
    """The logic for loading pretrain model.
  
     Args:
      MODEL_OBJ: Model object

    Returns:
      It returns keras model objcet

    """
    try:
       model =  config_model['MODEL_OBJ']
       print("Detected pretrain model!!")
       # os.makedirs('Models', exist_ok = True)
       # save_path = os.path.join('Models', config_model['MODEL_NAME']+'.h5')
       # print(f'Model has been saved following directory : {save_path}')
       # model.save(save_path)
       return model

    except Exception as e:
        print("Something went wrong!!", e)


def model_preparation(model):
    print('Preparing model...')
    if config_model['FREEZE_ALL'] == 'True':
        print('Freezing all...')
        for layer in model.layers:
            layer.trainable = False

        # Add custom layers
        flatten_in = tf.keras.layers.Flatten()(model.output)

        if config_data['CLASSES'] > 2:
            print('Adding softmax...')
            prediction = tf.keras.layers.Dense(
                units=config_data['CLASSES'],
                activation="softmax"
            )(flatten_in)

            full_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction
            )

            full_model.compile(
                optimizer=config_model['OPTIMIZER'],
                loss=config_model['LOSS_FUNC'],
                metrics=["accuracy"]
            )
            print('Model loaded!!')

            return full_model

        else:
            print('Adding sigmoid...')
            prediction = tf.keras.layers.Dense(
                units=config_data['CLASSES'],
                activation="sigmoid"
            )(flatten_in)

            full_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=prediction
            )

            full_model.compile(
                optimizer=config_model['OPTIMIZER'],
                loss=config_model['LOSS_FUNC'],
                metrics=["accuracy"]
            )

            print('Model loaded!!')

            return full_model


    else:

        for layer in model.layers[:config_model['FREEZE_TILL']]:
            layer.trainable = False

            # Add custom layers
            flatten_in = tf.keras.layers.Flatten()(model.output)

            if config_data['CLASSES'] > 2:
                prediction = tf.keras.layers.Dense(
                    units=config_data['CLASSES'],
                    activation="softmax"
                )(flatten_in)

                full_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=prediction
                )

                full_model.compile(
                    optimizer=config_model['OPTIMIZER'],
                    loss=config_model['LOSS_FUNC'],
                    metrics=["accuracy"]
                )
                print('Model loaded!!')

                return full_model

            else:
                prediction = tf.keras.layers.Dense(
                    units=config_data['CLASSES'],
                    activation="sigmoid"
                )(flatten_in)

                full_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=prediction
                )

                full_model.compile(
                    optimizer=config_model['OPTIMIZER'],
                    loss=config_model['LOSS_FUNC'],
                    metrics=["accuracy"]
                )

                print('Model loaded!!')

                return full_model



def load_pretrain_model():

    """The logic for loading pretrain model.
  
     Args:
      MODEL_OBJ: Model object

    Returns:
      It returns keras model objcet

    """
    model = get_model()
    model = model_preparation(model)
    return model


def load_exist_model():

    """The logic for loading an existing model.
  
     Args:
      PRETRAIN_MODEL_DIR: Your existing model path

    Returns:
      It returns keras model objcet

    """
    print('Loading existing model...')
    print("Model loaded!")
    model = tf.keras.models.load_model(config_model['PRETRAIN_MODEL_DIR'])
    model = model_preparation(model)
    return model










