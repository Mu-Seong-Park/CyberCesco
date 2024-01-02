from flask import Flask, request, abort
from flask_cors import CORS
from flask_socketio import SocketIO
from io import BytesIO
import numpy as np
import cv2

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
# CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB


def process_video(byte_data):
    # 가상의 파일 객체 생성
    video_file = BytesIO(byte_data)

    frames = []
    while True:
        image_data = video_file.read(1024)  # 적절한 버퍼 크기로 읽기
        if not image_data:
            break

        # NumPy 배열로 변환
        nparr = np.frombuffer(image_data, np.uint8)

        # OpenCV의 이미지로 디코딩
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames.append(frame)

    # 가상의 파일 객체를 닫습니다.
    video_file.close()

    return frames


@app.route('/upload', methods=['POST'])
def save_file():
    # 업로드된 파일을 저장할 디렉토리 경로
    upload_directory = 'C:/Users/xssds/OneDrive/Desktop/Study/temp/'

    # 바이트 데이터로 받아오기
    file_data = request.data

    if not file_data:
        print("NO DATA!!!!!")
        abort(400, "No data provided")

    # 임시로 데이터를 처리하는 부분은 아직 model 개발이 안 되어서 주석 처리해둠.##########################################
    # # 받은 바이트 데이터를 process_video 함수로 전달
    # result_frames = process_video(file_data)

    # 여기에서 result_frames를 활용하여 추가적인 처리 수행

    # # 파일 데이터를 저장하기 위해 파일을 열고 쓰기
    # with open(upload_directory + 'uploaded_file.mp4', 'wb') as file:
    #     file.write(file_data)
    print('socketio 호출')
    socketio.emit('videoCheck', {'result': 'Check Result'}, namespace='/video')
    print('emit 완료')

    return 'File uploaded successfully!'


@socketio.on('connect')
def handle_connect():
    print('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, debug=True)
