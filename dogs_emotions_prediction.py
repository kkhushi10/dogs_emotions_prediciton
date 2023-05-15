from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
input_shape = (224, 224, 3)
from tensorflow.keras.models import Model
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_data = train_datagen.flow_from_directory('/Users/khushi/Downloads', target_size=input_shape[:2], batch_size=32, class_mode='categorical')
val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory('/Users/khushi/Downloads',
                                            target_size=input_shape[:2],
                                            batch_size=32,
                                            class_mode='categorical')

model.fit(train_data,
          validation_data=val_data,
          epochs=10)


model.save('dog_emotions_model.h5')