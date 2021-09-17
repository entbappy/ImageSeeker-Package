import os
import sys
import warnings

from ImageSeeker.utils import models_config as mc
import json

"""Public ImageSeeker utilities.
This module is used as a shortcut to access all the symbols. Those symbols was
exposed under train engine and predict engine.
"""

with open('config.json','r') as f:
      params = json.load(f)

#Data config
TRAIN_DATA_DIR = params['TRAIN_DATA_DIR']
print(TRAIN_DATA_DIR)
VALID_DATA_DIR = params['VALID_DATA_DIR']
CLASSES = params['CLASSES']
SIZE = params['IMAGE_SIZE'].split(',')
h = int(SIZE[0])
w = int(SIZE[1])
c = int(SIZE[2])
IMAGE_SIZE = h,w,c
AUGMENTATION = params['AUGMENTATION']
BATCH_SIZE = params['BATCH_SIZE']

# Model config
MODEL_OBJ = params['MODEL_OBJ']
print("I am Model obj", MODEL_OBJ)
MODEL_OBJ = mc.return_model(MODEL_OBJ)
MODEL_NAME = params['MODEL_NAME']
EPOCHS = params['EPOCHS']
OPTIMIZER = params['OPTIMIZER']
LOSS_FUNC = params['LOSS_FUNC']
FREEZE_ALL = params['FREEZE_ALL']



def configureData(TRAIN_DATA_DIR = TRAIN_DATA_DIR, VALID_DATA_DIR = VALID_DATA_DIR, AUGMENTATION = AUGMENTATION, CLASSES = CLASSES, IMAGE_SIZE = IMAGE_SIZE, BATCH_SIZE = BATCH_SIZE):
    CONFIG = {
        'TRAIN_DATA_DIR' : TRAIN_DATA_DIR,
        'VALID_DATA_DIR' : VALID_DATA_DIR,
        'AUGMENTATION': AUGMENTATION,
        'CLASSES' : CLASSES,
        'IMAGE_SIZE' : IMAGE_SIZE,
        'BATCH_SIZE' : BATCH_SIZE,
    }

    return CONFIG





def configureModel(MODEL_OBJ = MODEL_OBJ, MODEL_NAME=MODEL_NAME, EPOCHS = EPOCHS, FREEZE_ALL= FREEZE_ALL , OPTIMIZER=OPTIMIZER, LOSS_FUNC=LOSS_FUNC):
    CONFIG = {
        'MODEL_OBJ' : MODEL_OBJ,
        'MODEL_NAME' : MODEL_NAME,
        'EPOCHS' : EPOCHS,
        'FREEZE_ALL' : FREEZE_ALL,
        'OPTIMIZER': OPTIMIZER,
        'LOSS_FUNC' : LOSS_FUNC,
    }

    return CONFIG