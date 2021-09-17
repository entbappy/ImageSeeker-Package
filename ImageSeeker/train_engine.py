'''
@author: Bappy Ahmed
Email: entbappy73@gmail.com
Date: 06-sep-2021
'''



from ImageSeeker.utils import model
from ImageSeeker.utils import data_manager as dm
from ImageSeeker.utils.config import configureModel
from ImageSeeker.utils import callbacks
import tensorflow as tf

config_model = configureModel()


def train():


    """The logic for one training step.
  
    This method should contain the mathematical logic for one step of training.
    This typically includes the forward pass, loss calculation, backpropagation,
    and metric updates.

     Args:
      data: A nested structure of `Tensor`s.

    Returns:
      A `dict` containing values.Typically, the
      values of the `Model`'s metrics are returned. Example:
      `{'loss': 0.2, 'accuracy': 0.7}`.

    """
    model_obj = model.load_pretrain_model()
    my_model = model_obj
    train_data, valid_data = dm.train_valid_generator()

    #callbacks
    log_dir = callbacks.get_log_path()
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    ckp = callbacks.checkpoint()

    call = [tb_cb, ckp]

    #Calculating steps_per_epoch & validation_steps
    steps_per_epoch = train_data.samples // train_data.batch_size
    validation_steps = valid_data.samples // valid_data.batch_size

    my_model.fit(
        train_data,
        validation_data=valid_data,
        epochs=config_model['EPOCHS'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=call
    )

    new_path = f"New_trained_model/{'new'+config_model['MODEL_NAME']+'.h5'}"
    my_model.save(new_path)
    print(f"Model saved at the following location : {new_path}")
