import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# 데이터 경로 설정
data_dir = './train'  # 이미지 데이터 폴더 경로
img_width, img_height = 150, 150  # 이미지 크기
input_shape = (img_width, img_height, 3)
epochs = 20
batch_size = 32

# 데이터 생성기 설정
datagen = ImageDataGenerator(
    rescale=1. / 255,  # 픽셀 값을 0과 1 사이로 조정
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2)  # 20%를 검증 데이터로 사용

# 데이터 로드 및 분할
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')  # 학습용 데이터

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')  # 검증용 데이터

# CNN 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# 모델 학습
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)


# 모델 저장
model.save('cat_dog_classifier.h5')

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