
#라이브러리 불러오기 
import os
from turtle import distance 
import numpy as np 
from PIL import Image 
import faiss 
import face_recognition
import cv2

gWait = False


################################################
# 함수 리스트
################################################
# 많이 나온 단어 확인
def most_frequent(data):
    countList = [] 
    for x in data:
        countList.append(data.count(x))

    return data[countList.index(max(countList))], max(countList)

################################################

################################################

#백터 DB 불러오기 
faceIndex = faiss.read_index('./web/train/face_20240527.bin')  #학습결과

trainLabels = np.load('./web/train/labels.npy') #학습결과 정답(라벨)


def face_detect(imageData):
    global gWait

    if gWait == True:
        return "unknown"

    gWait = True
    pilImg = Image.fromarray(imageData)
    image = pilImg.save('./web/train/test.jpg')

    #얼굴 인식:
    testImage = face_recognition.load_image_file('./web/train/test.jpg')
    testFace = face_recognition.face_locations(testImage)

    if len(testFace) != 1:
        gWait = False
        return "unknown"

    top, right, bottom, left = testFace[0]
    # print(top, right, bottom, left)

    faceCut = testImage[top-20:bottom+20, left-20:right+20]

    ########### 버그 함찬 고쳤던 부분 ###########
    pilImage = Image.fromarray(faceCut)
    pilImage.save('./test.jpg')
    img = face_recognition.load_image_file('./test.jpg')
    ############################################# 


    #encoding
    testEncoding = face_recognition.face_encodings(img)[0]
    testEncoding = np.array(testEncoding, dtype= np.float32).reshape(-1,128)
    # (128) -> (1,128)

    #예측:
    distance, result = faceIndex.search(testEncoding, k = 5)

    #값 검출 
    label = [trainLabels[i] for i in result[0]]

    faceResult = most_frequent(label)

    gWait = False

    # if faceResult[1] < 2:
    #     return "unknown"
    # else:
    #     return faceResult[0]
    return faceResult[0]




'''
train되지 않은 얼굴들을 예측하라고 주어졌을 때, 있는 라벨 중 그나마 가장 근접한 인물을 알려준다. 
이것을 방지하기 위해 안 닮은 사람들을 unknown 처리를 해야 한다. 


이때 faiss 최소 count을 정해, 일정 기준보다 먼 데이터이면 unknown으로 해줘야 할 것이다.

distance [[0.13162771 0.13510479 0.13760538 0.14405231 0.16313717]]
the final label: ['jungwoosung', 'jungwoosung', 'jungwoosung', 'jungwoosung', 'jungwoosung']

distance [[0.11258335 0.12386454 0.13615938 0.14047796 0.17761636]]
the final label: ['eomjiyoon', 'eomjiyoon', 'eomjiyoon', 'eomjiyoon', 'eomjiyoon']



최종 인물 알려주는 기준: 
정해 놓은 K nearest neighbors 중, n개 이상 같은 인물로 측정됐을 시 결과로 정한다. 
테스트:
100명 이내 train한 경우 4개정도가 충분.  
100명 이상일 때, 3개 정도가 좋았다. 


얼굴 인식은 어떤 문제를 해결하고 싶은지 따라 true positive에 중점을 둘 것인지 혹은 false positive을 줄일 것인지 선택해야 한다.
만약 얼굴 인식을 통해 보안관련 프로그램을 만든다고 하면, false positive를 줄이는 방향이 맞을 것으로 보이며, 얼굴을 인식해서 미용 관련 
서비스를 운영한다 하면 true positive에 집중해도 될 것이라고 생각했다. 

'''