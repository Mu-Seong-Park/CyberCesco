import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np

# 모델 불러오기
model = tf.keras.models.load_model('finger_classification_model.h5')

# 이미지 파일 경로
image_path = './train/class0/2.png'

# 이미지 읽기
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 이미지 전처리
img = cv2.resize(img, (224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# 모델 예측
predictions = model.predict(img_array)
print(predictions[0][0])
# 이진 분류에서는 간단하게 0 또는 1로 변환
prediction_result = 1 if predictions[0][0] > 0.5 else 0

# 결과 출력
class_label = 'hand' if prediction_result == 1 else 'non-hand'
print(f"Prediction: {class_label}")

# 검출된 경우에는 결과 이미지 저장
if prediction_result == 1:
    # 결과 이미지를 저장할 경로
    output_path = 'path/to/your/output/result_image.jpg'

    # 결과 이미지 저장
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Result image saved at {output_path}")