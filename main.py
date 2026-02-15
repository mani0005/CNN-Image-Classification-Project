import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

images = '/Users/apple/Downloads/archive/myntradataset/images'
styles_df = pd.read_csv("/Users/apple/Downloads/archive/myntradataset/styles.csv", on_bad_lines='skip')

styles_df.head()

styles_df['filename'] = styles_df['id'].astype(str) + '.jpg'

styles_df

label_encoder = LabelEncoder()
styles_df['encoded_labels'] = label_encoder.fit_transform(styles_df['articleType'])

train_df, val_df = train_test_split(styles_df, test_size=0.2, random_state=42)
train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)

# Data augmentation and normalization
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=images,  
    x_col="filename",
    y_col="encoded_labels",
    target_size=(150, 150), 
    batch_size=32,
    class_mode="raw" 
)

validation_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=images,
    x_col="filename",
    y_col="encoded_labels",
    target_size=(150, 150),
    batch_size=32,
    class_mode="raw"
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=images,
    x_col="filename",
    y_col="encoded_labels",
    target_size=(150, 150),
    batch_size=32,
    class_mode="raw"
)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the model
model.save("model.h5")

# Save the label encoder classes
np.save("label.npy", label_encoder.classes_)