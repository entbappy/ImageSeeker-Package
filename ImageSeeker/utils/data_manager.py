from ImageSeeker.utils.config import configureData
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

config = configureData()  #Defining the configureData object

def train_valid_generator():

    '''
    This function generates train & valid data from images path,
    it also performs data augmentation.
    :return object of data:
    '''

    if config['AUGMENTATION'] == 'True':
        print("Augmetation applied!")
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)

        valid_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(
            directory= config['TRAIN_DATA_DIR'],
            target_size=config['IMAGE_SIZE'][:-1],
            batch_size=config['BATCH_SIZE'],
            class_mode='categorical')

        valid_set = valid_datagen.flow_from_directory(
            directory=config['VALID_DATA_DIR'],
            target_size=config['IMAGE_SIZE'][:-1],
            batch_size=config['BATCH_SIZE'],
            class_mode='categorical')

        return training_set, valid_set

    else:
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        valid_datagen = ImageDataGenerator(rescale=1. / 255)

        training_set = train_datagen.flow_from_directory(
            directory=config['TRAIN_DATA_DIR'],
            target_size=config['IMAGE_SIZE'][:-1],
            batch_size=config['BATCH_SIZE'],
            class_mode='categorical')

        valid_set = valid_datagen.flow_from_directory(
            directory=config['VALID_DATA_DIR'],
            target_size=config['IMAGE_SIZE'][:-1],
            batch_size=config['BATCH_SIZE'],
            class_mode='categorical')

        return training_set, valid_set


def class_name():
    train, valid = train_valid_generator()
    print(train.class_indices)




def manage_input_data(INPUT_IMAGE):
    '''
    This function takes an new raw image & convert the image
    into nd array with respect to training images dimention.
    :param INPUT_IMAGE:
    :return nd array:
    '''
    try:
        INPUT_IMAGE = INPUT_IMAGE
        test_image = image.load_img(INPUT_IMAGE, target_size= config['IMAGE_SIZE'][:-1])
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        return test_image
    except Exception as e:
        print(e)