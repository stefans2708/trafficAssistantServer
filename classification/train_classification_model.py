import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG_PATH = 'C:\\Users\\stefa\\Desktop\\cars196\\dataset'
IMAGE_SIZE = (192, 192)
INPUT_SHAPE = IMAGE_SIZE + (3,)
AUTO_TUNE = tf.data.experimental.AUTOTUNE
LEARNING_RATE = 0.0001


def load_labels():
    dataset_path = os.path.join(IMG_PATH)
    return os.listdir(dataset_path)


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.prefetch(buffer_size=AUTO_TUNE)
    return ds


labels = load_labels()

train_dataset = image_dataset_from_directory(directory=IMG_PATH, image_size=IMAGE_SIZE, class_names=labels,
                                             validation_split=0.5, seed=123, subset='training')

validation_dataset = image_dataset_from_directory(directory=IMG_PATH, image_size=IMAGE_SIZE, class_names=labels,
                                                  validation_split=0.5, seed=123, subset='validation')

train_dataset = configure_for_performance(train_dataset)
validation_dataset = configure_for_performance(validation_dataset)

# Data Augmentation
data_augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=INPUT_SHAPE),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)
])

base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)
base_model.trainable = False

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(196, activation='softmax', trainable=True)

inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)
x = data_augmentation_layers(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

feature_extraction_history = model.fit(x=train_dataset,
                                       epochs=48,
                                       validation_data=validation_dataset,
                                       verbose=2)

# Fine tuning
num_of_fine_tuning_layers = 62
num_of_epochs = 37
total_epochs = len(feature_extraction_history.epoch) + num_of_epochs

base_model.trainable = True
for layer in base_model.layers[:-num_of_fine_tuning_layers]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE / 10),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

fine_tuning_history = model.fit(x=train_dataset,
                                epochs=total_epochs,
                                initial_epoch=feature_extraction_history.epoch[-1],
                                validation_data=validation_dataset)

# model.save('model') bug in tf 2.7

model.save('classification_model_4.h5')
