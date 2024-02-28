import cv2
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# 모델 불러오기
model = tf.keras.models.load_model('finger_classification_model.h5')

# 동영상 경로 또는 URL
video_path = 'C:/Users/qkdkr/Desktop/study/testdata1.mp4'

# 동영상 열기
cap = cv2.VideoCapture(video_path)

# 저장할 이미지 경로
output_folder = './output/'

# 프레임 번호 초기화
frame_count = 0

# 일정 간격으로 프레임을 건너뛰기 위한 변수
skip_frames = 30

# 프레임 단위로 동영상을 읽어서 처리
while True:
    # 일정 간격으로 프레임 건너뛰기
    for _ in range(skip_frames):
        ret, _ = cap.read()

    # 현재 프레임 읽기
    ret, frame = cap.read()

    # 동영상의 끝에 도달하면 종료
    if not ret:
        break

    # 프레임 전처리
    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # 모델 예측
    predictions = model.predict(img_array)

    # 이진 분류에서는 간단하게 0 또는 1로 변환
    prediction_result = 1 if predictions[0][0] > 0.5 else 0

    # 결과를 화면에 표시
    class_label = 'hand' if prediction_result == 1 else 'non-hand'
    cv2.putText(frame, f"Prediction: {class_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 손모양이 발견된 경우 프레임 저장
    if prediction_result == 1:
        frame_count += 1
        save_path = f"{output_folder}hand_frame_{frame_count}.png"
        cv2.imwrite(save_path, frame)
        print(f"Hand detected at frame {frame_count}. Image saved at {save_path}")

    # 화면에 출력
    cv2.imshow('Video', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()