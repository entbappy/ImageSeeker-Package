import os
from ImageSeeker.utils.config import configureModel
import time
import tensorflow as tf

config_model = configureModel()


"""Configures callbacks for use in various training loops.
  Args:
      get_log_path: unique log location.
      checkpoint : None
  Returns:
      Instance of CallbackList used to control all Callbacks.
 """


def get_log_path(log_dir="Tensorboard/logs/fit"):
  fileName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
  logs_path = os.path.join(log_dir, fileName)
  print(f"Saving logs at {logs_path}")
  return logs_path


def checkpoint():
    CKPT_path = "Checkpoint/Model_ckpt.h5"
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)
    return  checkpointing_cb

