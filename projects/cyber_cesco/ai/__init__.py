import os
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src import layers
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import matplotlib as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 데이터 경로 설정
data_dir = '.\\train'  # 이미지 데이터 폴더 경로
#
# # file 이름을 label과 id로 나누기.
# full_name = os.listdir(data_dir)
# labels = [each.split('.')[0] for each in full_name]
# file_id = [each.split('.')[1] for each in full_name]
# print(set(labels), len(file_id))

imageGenerator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.2, 1.3],
    horizontal_flip=True,
    validation_split=.2
)

trainGen = imageGenerator.flow_from_directory(
    os.path.join(data_dir, 'train_set'),
    target_size=(64, 64),
    subset='training'
)

validationGen = imageGenerator.flow_from_directory(
    os.path.join(data_dir, 'train_set'),
    target_size=(64, 64),
    subset='validation'
)

model = Sequential()

model.add(layers.InputLayer(input_shape=(64, 64, 3)))
model.add(layers.Conv2D(16, (3, 3), (1, 1), 'same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(32, (3, 3), (1, 1), 'same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Conv2D(64, (3, 3), (1, 1), 'same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(rate=0.3))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['acc'],
)

epochs = 50
history = model.fit(
    trainGen,
    epochs=epochs,
    validation_data=validationGen,
)

history.history.keys()

history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(loss)+1)
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(epochs, loss, color='blue', label='train_loss')
ax1.plot(epochs, val_loss, color='red', label='val_loss')
ax1.set_title('Train ans Validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('loss')
ax1.grid()
ax1.legend()

accuracy = history_dict['acc']
val_accuracy = history_dict['val_acc']

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(epochs, accuracy, color='blue', label='train_accuracy')
ax2.plot(epochs, val_accuracy, color='red', label='val_accuracy')
ax2.set_title('Train ans Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.grid()
ax2.legend()

plt.show()

testGenerator = ImageDataGenerator(
    rescale=1./255
)

testGen = imageGenerator.flow_from_directory(
    os.path.join(data_dir, 'test_set'),
    target_size=(64, 64),
)

model.save('cat_dog_model.h5')



# import tensorflow as tf
# from tensorflow import keras
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# print(tf.__version__);
# # 데이터셋 경로
# train_data_dir = './train'
# validation_data_dir = './validation'
#
# # 하이퍼파라미터
# img_width, img_height = 224, 224
# batch_size = 32
# epochs = 10
#
# # 데이터 전처리 및 증강
# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     rescale=1. / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True
# )
#
# validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
#
# train_generator = train_datagen.flow_from_directory(
#     train_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary'
# )
#
# validation_generator = validation_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode='binary'
# )
#
# # ResNet50 모델 불러오기 (사전 훈련된 가중치 사용)
# base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
#
# # 모델 구성
# model = keras.models.Sequential()
# model.add(base_model)
# model.add(keras.layers.GlobalAveragePooling2D())
# model.add(keras.layers.Dense(256, activation='relu'))
# model.add(keras.layers.Dropout(0.5))
# model.add(keras.layers.Dense(1, activation='sigmoid'))  # Change softmax to sigmoid for binary classification
#
# # 사전 훈련된 레이어 동결
# for layer in base_model.layers:
#     layer.trainable = False
#
# # 모델 컴파일
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Change categorical_crossentropy to binary_crossentropy
#
# # 모델 학습
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size
# )
#
# # 모델 저장
# model.save('finger_classification_model')