# Muhammad Anees
# 537630

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load and Preprocess the Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Feature Extraction with VGG16
base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
for layer in base_model_vgg.layers:  # Fixed: changed base_model to base_model_vgg
    layer.trainable = False

x = Flatten()(base_model_vgg.output)  # Fixed: changed base_model to base_model_vgg
x = Dense(512, activation='relu')(x)
output_vgg = Dense(10, activation='softmax')(x)  # Renamed for clarity

model_feature_extraction = Model(inputs=base_model_vgg.input, outputs=output_vgg)  # Fixed variables
model_feature_extraction.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_feature_extraction = model_feature_extraction.fit(
    x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=32
)

# Fine-Tuning with ResNet50
base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))  # Use distinct name
for layer in base_model_resnet.layers:
    layer.trainable = False

x = Flatten()(base_model_resnet.output)
x = Dense(512, activation='relu')(x)
output_res = Dense(10, activation='softmax')(x)

model_finetune = Model(inputs=base_model_resnet.input, outputs=output_res)  # Fixed: used output_res
model_finetune.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Unfreeze the last 5 layers of ResNet50
for layer in base_model_resnet.layers[-5:]:
    layer.trainable = True

model_finetune.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

history_finetune = model_finetune.fit(
    x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=32
)

# Evaluate Both Models
loss_feature_extraction, acc_feature_extraction = model_feature_extraction.evaluate(x_test, y_test)
loss_finetune, acc_finetune = model_finetune.evaluate(x_test, y_test)

# Compare Results
print(f"Feature Extraction Accuracy: {acc_feature_extraction * 100:.2f}%")
print(f"Fine-Tuning Accuracy: {acc_finetune * 100:.2f}%")
