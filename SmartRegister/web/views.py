from http.client import HTTPResponse
from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse
import cv2
from .facedetector import FaceDetector
from .models import RegisterUser
from .faiss_predict import face_detect

# Create your views here.


def index(request):
    dbList = "This is an initial index page"
    return render(
        request,
        'web_index.html',
        {'data': dbList}
    )

########## cctv_main #######################


def cctv(request):
    return render(
        request,
        'cctv_main.html',
    )


def video_feed(request):
    return StreamingHttpResponse(
        stream(),
        content_type='multipart/x-mixed-replace; boundary=frame')


def stream():
    predictorPath = "web/shape_predictor_68_face_landmarks.dat"

    faceDetect = FaceDetector(
        detectorPath="", predictorpath= predictorPath)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("cant find the camera")
            break
    

        faceDetect.load_image(frame)
        faceCount = faceDetect.detect_faces()

        if len(faceCount) == 1:
            predictedName = face_detect(frame) 
            # cv2.putText(frame, predictedName, (50,50), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color = (255,0,0),thickness = 3, lineType = cv2.LINE_AA )

            if predictedName != "unknown":
                addUser = RegisterUser()
                addUser.names = predictedName 
                addUser.telnos = "0000000000"
                addUser.save()

        # faceDetect.draw_landmark()
        # face_detect(frame)
        # face_detector.draw_faces()

        # 서버로 데이터 전송하기 위해 이미지를 binary로 변형
        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-type:image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

        

    cap.release()
    cv2.destroyAllWindows()

#################################


def add_user(request):
    #데이터베이스에 자료 저장 
    addUser = RegisterUser()
    addUser.names = '홀길동'
    addUser.telnos = '010-0000-0000'
    addUser.save()
    return HttpResponse("adduser 등록")


def face_test(request):
    result = face_detect('./static/test.jpg')
    print(result)
    return HttpResponse("face test")



def delete_user(request):
    try:
        user_to_delete = RegisterUser.objects.get(id=id)
        user_to_delete.delete()
        return HttpResponse("User with name '홀길동' deleted successfully.")
    except RegisterUser.DoesNotExist:
        return HttpResponse("User with name '홀길동' does not exist.")