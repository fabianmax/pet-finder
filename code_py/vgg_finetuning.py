import pandas as pd
import numpy as np
import os, fnmatch
import datetime
import matplotlib.pyplot as plt
import json

import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

# Check for GPUs on GC
K.tensorflow_backend._get_available_gpus()

# Load data
df_train = pd.read_pickle('data/prepared/train.pkl')

# Settings
BATCH_SIZE = 64

# Path and files
train_images_path = '/home/fabianmueller/pet_finder/data/raw/images/train_images'
train_images = fnmatch.filter(os.listdir(train_images_path), '*.jpg')

# Make binary target
target = 'adoptionspeed_bin'
df_train.loc[:, target] = np.where(df_train.loc[:, 'adoptionspeed'] <= 2, 'adopted', 'not_adopted')

# Create dict with id and target class
name_target_dict = df_train.set_index('petid')['adoptionspeed_bin'].to_dict()

# Empty dict for generator
dict_generator = {'filename': [], 'class': []}

# Go through images and save in generator dict
for name in train_images:
    short_name = name.split('-')[0]
    label = name_target_dict[short_name]

    dict_generator['filename'].append(name)
    dict_generator['class'].append(label)

# Convert dict to pd.DataFrame
df_generator = pd.DataFrame(dict_generator)

# Create train/valid DataFrames
df_generator_train = df_generator.sample(frac=0.7)
df_generator_valid = df_generator.drop(df_generator_train.index)

# Create generator with rescaling
data_gen = ImageDataGenerator(rescale=1. / 255)

# Training generator
train_generator = data_gen.flow_from_dataframe(dataframe=df_generator_train,
                                               directory=train_images_path,
                                               x_col='filename',
                                               y_col='class',
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               shuffle=True,
                                               batch_size=BATCH_SIZE)

# Validation generator
valid_generator = data_gen.flow_from_dataframe(dataframe=df_generator_valid,
                                               directory=train_images_path,
                                               x_col='filename',
                                               y_col='class',
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               shuffle=True,
                                               batch_size=BATCH_SIZE)

# Number of steps per echo
STEPS_PER_ECHO_TRAIN = np.ceil(train_generator.n / BATCH_SIZE)
STEPS_PER_ECHO_VALID = np.ceil(valid_generator.n / BATCH_SIZE)

# Number of steps per echo
STEPS_PER_ECHO = np.ceil(train_generator.n / BATCH_SIZE)

# VGG model including imagenet weights
model = VGG16(include_top=False,
              weights='imagenet',
              input_shape=(224, 224, 3))

# Fix all layers
for layer in model.layers:
    layer.trainable = False

# Adding new dense and output layer (simple version)
x = model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
p = Dense(2, activation="softmax")(x)

# Creating the final model
model_final = Model(input=model.input, output=p)

# Compile the model
model_final.compile(loss="categorical_crossentropy",
                    optimizer=Adam(0.0001),
                    metrics=["accuracy"])

# Checkpointing
filepath = "checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5" 
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# Earlystopping
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10)

# Combine callbacks
callbacks_list = [checkpoint, earlystopping]

# Fit model (hopefully earlystopping will kick in)
hist = model_final.fit_generator(train_generator,
                                 epochs=100,
                                 steps_per_epoch=STEPS_PER_ECHO_TRAIN,
                                 validation_data=valid_generator,
                                 validation_steps=STEPS_PER_ECHO_VALID,
                                 callbacks=callbacks_list)


# Save model
name = datetime.datetime.today().strftime('%Y-%m-%d')
model_final.save(name + '_model_final.h5')

# Save history
with open(name + 'hist_final.json', 'w') as f:
    json.dump(hist.history, f)


# Plot learning history
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
