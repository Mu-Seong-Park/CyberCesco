from flask import Flask, request, abort
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB


@app.route('/upload', methods=['POST'])
def save_file():
    # 업로드된 파일을 저장할 디렉토리 경로
    upload_directory = 'C:/Users/xssds/OneDrive/Desktop/Study/temp/'

    # 바이트 데이터로 받아오기
    file_data = request.data

    if not file_data:
        print("NO DATA!!!!!")
        abort(400, "No data provided")

    # 파일 데이터를 저장하기 위해 파일을 열고 쓰기
    with open(upload_directory + 'uploaded_file.mp4', 'wb') as file:
        file.write(file_data)

    return 'File uploaded successfully!'


if __name__ == '__main__':
    app.run(debug=True)