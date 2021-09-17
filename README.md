
# ImageSeeker: An Image Classification Library

  
![ImageSeeker Logo](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/ImageSeeker.png)

    


This repository hosts the development of the ImageSeeker library.

  
## Authors

- [@Bappy Ahmed](https://www.linkedin.com/in/boktiarahmed73/)
Email: entbappy73@gmail.com

Github: https://github.com/entbappy/
  
## 🚀 About Me
I'm a Data Science learner. This library I have implemented just for learning purpose.

  
## About ImageSeeker

ImageSeeker is a deep learning image classification library written in Python, running on top of the machine learning platform TensorFlow.Keras. It was developed with a focus on enabling fast experimentation of images classification. You can classify any image with any classification model in Keras appliaction by just writing 4 lines of code.


ImageSeeker is:

- **Simple** -- ImageSeeker reduces developer time of writing too much code of any image classification problem so that they focus on the parts of the problem that really matter.
- **Flexible** -- Keras adopts the principle of *progressive disclosure of complexity*: simple workflows should be quick and easy that’s why ImageSeeker is Flexible.
- **Powerful** -- Keras provides industry-strength performance and scalability: so we can use ImageSeeker in the production

## Keras & ImageSeeker 

Keras is the high-level API of TensorFlow 2: an approachable, highly-productive interface
for solving machine learning problems, with a focus on modern deep learning. We can do so many tasks using Keras and image classification is one of them, but the issue is like we need to write many lines of code to implement an image classification solution.

But in ImageSeeker you don’t need to write many lines of code for implementing an image classification solution. You don’t also have to worry about your data preparation. What you need to do is just have to define your data path & some of the parameters of the model yes, your work will be done!


## Image Classification using Keras

Necessary library importing
```python
from tensorflow.keras.applications.densenet import VGG16
import tensorflow.keras as keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os


```

Redirecting to the data path

```python

ROOT = 'H:\\Parsonal\\Coding Practice\\dogCat'
os.chdir(ROOT)
os.getcwd()


```
Preparing data & applying augmentation

```python

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale= 1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip= True)

test_datagen = ImageDataGenerator(rescale= 1./255)

Train_path = "H:\\Parsonal\\Coding Practice\\dogCat\\train"
Test_path = "H:\\Parsonal\\Coding Practice\\dogCat\\valid"

training_set = train_datagen.flow_from_directory(directory=Train_path,
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                 class_mode='categorical')


test_set = test_datagen.flow_from_directory(directory=Test_path,
                                                 target_size=(224,224),
                                                 batch_size=32,
                                                 class_mode='categorical')



```


Dowloading models

```python

VGG = tf.keras.applications.VGG16(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)

)

VGG.summary()

```

Freezing layers

```python

for layer in VGG.layers:
    layer.trainable = False

```

Adding custom layers

```python

model = models.Sequential()
model.add(VGG)               
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu' ))  
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax')) 
#model.summary()

```
Defining optimizers and loss function

```python

from tensorflow.keras.optimizers import RMSprop
loss = 'categorical_crossentropy'
optimizer =RMSprop(learning_rate=0.0001)


```
Defining Tensorboard log and checkpoint

```python

# Log

import time 

def get_log_path(log_dir="logs/fit"):
  fileName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
  logs_path = os.path.join(log_dir, fileName)
  print(f"Saving logs at {logs_path}")
  return logs_path

log_dir = get_log_path()
tb_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


# checkpoint

CKPT_path = "Model_ckpt.h5"
checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)


```

Compiling model

```python

model.compile(optimizer = optimizer, loss=loss, metrics=['accuracy'])

```

Start training

```python

hist = model.fit(training_set,
                         steps_per_epoch = 10,
                         epochs = 5,
                         validation_data = test_set,    
                         validation_steps = 20,
                         callbacks=[tb_cb,checkpointing_cb]
                         )


```

 Epoch 1/5
10/10 [==============================] - 231s 21s/step - loss: 3.4007 - accuracy: 0.3812 - val_loss: 0.8920 - val_accuracy: 0.6438
Epoch 2/5
10/10 [==============================] - 100s 10s/step - loss: 1.0077 - accuracy: 0.6375 - val_loss: 0.6927 - val_accuracy: 0.7328
Epoch 3/5
10/10 [==============================] - 83s 9s/step - loss: 0.6629 - accuracy: 0.7344 - val_loss: 0.4763 - val_accuracy: 0.8266
Epoch 4/5
10/10 [==============================] - 69s 7s/step - loss: 0.4356 - accuracy: 0.8406 - val_loss: 0.6980 - val_accuracy: 0.7859
Epoch 5/5
10/10 [==============================] - 64s 6s/step - loss: 0.7226 - accuracy: 0.7844 - val_loss: 0.3199 - val_accuracy: 0.8906

#### Note: Then again you need write code for prediction as well


# Image Classification using ImageSeeker:

## >>Installation / Setup of ImageSeeker
**First of all create a new conda environment and activate the environment:**
```bash
 conda create -n imageseeker python==3.7.10 -y
```
```bash
 conda activate imageseeker
```
**Great! After that install ImageSeeker from pypi using that command:**

```bash
 pip install ImageSeeker
```

If you done the installation then from your terminal just write **imgseeker** keyword it will automatically launch the UI on your local host.


## Step-1:

![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/1.png)

  

## Launching UI on local host:

![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/2.png)


Yes!! You have got 😀 Now if you scroll down, you will see how you can get start with ImageSeeker below

## Step-2:
**Click on GET STARTED**

 ![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/3.png) 


After clicking on that you will get a form like that and you have to configure it for training 

![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/4.png) 

## Step-3:
**Configure those parameters: Example**

![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/7.png) 


Now just hit **Start Trainig** your training will be started And you can see all the log from your terminal itself like that.

![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/9.png) 


Detected pretrain model!!

Model has been saved following directory : Models\VGG16.h5
Preparing model...

Freezing all...

Adding sigmoid...

Model loaded!!

Augmetation applied!

Found 200 images belonging to 2 classes.

Found 110 images belonging to 2 classes.

Saving logs at Tensorboard/logs/fit\log_2021_09_09_02_24_11
Epoch 1/5
6/6 [==============================] - 35s 6s/step - loss: 6.3008 - accuracy: 0.5385 - val_loss: 1.5761 - val_accuracy: 0.5104
Epoch 2/5
6/6 [==============================] - 29s 5s/step - loss: 2.8433 - accuracy: 0.5109 - val_loss: 2.4668 - val_accuracy: 0.5312
Epoch 3/5
6/6 [==============================] - 29s 5s/step - loss: 2.3428 - accuracy: 0.4800 - val_loss: 0.8409 - val_accuracy: 0.5417
Epoch 4/5
6/6 [==============================] - 29s 5s/step - loss: 0.8796 - accuracy: 0.5927 - val_loss: 0.6251 - val_accuracy: 0.6667
Epoch 5/5
6/6 [==============================] - 29s 5s/step - loss: 0.6290 - accuracy: 0.6746 - val_loss: 0.5840 - val_accuracy: 0.7188
Model saved at the following location : New_trained_model/newVGG16.h5


## Step-4:
**Prediction:**

Awsome!! 😀😀 You have done the training. Now it's time for predicting. You see there is a button called **Predict**

![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/6.png) 


If you hit that button then you will get your prediction UI like that. Now you can upload images & start predicting😎

![Example](https://raw.githubusercontent.com/entbappy/Branching-tutorial/master/New%20folder%20(2)/10.png) 


# Congratulations !! 😀✌ You are done 😊

Yes! that's how it is simple.





## Acknowledgements

 - [Keras](https://keras.io/)
 - [SUNNY BHAVEEN CHANDRA](https://www.linkedin.com/in/c17hawke/)
 - [Sourangshu Pal](https://www.linkedin.com/in/sourangshu-pal-0774b212a/)
 - [Python](https://docs.python.org/3/)

  