from flask import Flask, request, abort, jsonify
from flask_socketio import SocketIO
from io import BytesIO
import numpy as np
import cv2
import requests
import multiprocessing
import time


app = Flask(__name__)
socketio = SocketIO(app, namespace='/socket.io', cors_allowed_origins="*")
# CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB


async def process_video(byte_data):
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


def temp_data_check(byte_data,auth):
    print("비동기 작동중......")
    #가상의 파일 객체 생성.
    video_file = BytesIO(byte_data)

    # spring으로 보내줄 JSON 더미 데이터, result_key 부분은 나중에 분 초나 몇번째 프레임인지
    # 이런 것으로 판별하거나, 이미지 등으로 대체할 예정.
    response = {"Auth": auth, "result_key": ["123"]}


    i = 10000
    while True:
        if i == 0:
            break
        else:
            print(i)
            i = i - 1
    print("전송 시작.")

    send_classification_results(auth)


def send_classification_results(classified_results):
    print("전송 메서드 진입.......")
    # 다른 서버의 '/event' 엔드포인트로 분류 결과를 전송
    results = {'results': "complete", 'auth': classified_results}
    url = 'http://localhost:8080/videoResult'
    response = requests.post(url, json=results)
    if response.status_code == 200:
        print('Classification results successfully sent to the event endpoint')
    else:
        print('Failed to send classification results to the event endpoint')

@app.route('/upload', methods=['POST'])
def check_file():
    # 업로드된 파일을 저장할 디렉토리 경로
    upload_directory = 'C:/Users/xssds/OneDrive/Desktop/Study/temp/'

    #header의 token 정보 읽어오기
    authorization_header = request.headers.get('Authorization')

    if authorization_header:
        # Authorization 헤더가 존재할 경우 처리
        print(request.headers)
    else:
        # Authorization 헤더가 존재하지 않을 경우 처리
        print('Authorization Header not found')
        return 'Authorization does not Exist!!'
    # 바이트 데이터로 받아오기
    file_data = request.data

    if not file_data:
        print("NO DATA!!!!!")
        abort(400, "No data provided")

    # 임시로 데이터를 처리하는 부분은 아직 model 개발이 안 되어서 주석 처리해둠.##########################################
    # # 받은 바이트 데이터를 process_video 함수로 전달
    # result_frames = process_video(file_data)

    # 여기에서 result_frames를 활용하여 추가적인 처리 수행

    # 멀티 프로세싱을 통해서 나중에 검사 결과를 전송할 수 있도록 메서드 호출.
    multiprocessing.Process(target=temp_data_check, args=(file_data, authorization_header)).start()

    print("메인 메서드 작동중........")
    # JSON 형식으로 반환.
    return jsonify({'result': 'transfer complete'})


if __name__ == '__main__':
    socketio.run(app, debug=True)
