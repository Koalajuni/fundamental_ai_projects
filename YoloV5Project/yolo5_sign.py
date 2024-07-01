#라이브러리 불러오기 
from unittest import result
import cv2
import yolov5 
import time
import torch 
import pandas as pandas
import yaml


#yolov5 Custom Model 불러오는 방법 
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path= './train/best.pt',
    force_reload= True
)


#yaml에 있는 데이터셋(labels) 불러오기 

data_yaml = './train/data.yaml'

with open(data_yaml) as f:
    data = yaml.load(f, Loader = yaml.FullLoader)

# print(data)
# print(data['names'])


labels = data['names']
# print(labels)



cap = cv2.VideoCapture(1)

cap.set(3,1280)
cap.set(4,720)


while True:
    _, frame = cap.read()

    frame = cv2.flip(frame,1)

    #yolov5 처리 이후 화면 보여주기 
    results = model(frame)
    pdTable = results.pandas().xyxy[0]
    pdList = pdTable.values.tolist()

    label = results.xyxy[0][:,-1].cpu().numpy()
    boundingBox = results.xyxy[0][:,:-2].cpu().numpy()
    score = results.xyxy[0][:,-2].cpu().numpy()

    # print(f"labels:  {labels}")
    # print(f"boundingBox {boundingBox}")
    # print(f"score: {score}")

    # percent 구하기 
    if len(score) > 0:
        confScore = round(score[0] * 100, 2)
        printScore = f'  {confScore}%'


    # 바운딩박스 처리 
    if len(label) > 0: 
        x1,y1,x2,y2 = int(boundingBox[0][0]), int(boundingBox[0][1]), int(boundingBox[0][2]),int(boundingBox[0][3])   
        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),1)
        cv2.rectangle(frame,(x1,y1-30),(x1+190,y1),(0,0,255),-1)
        cv2.putText(frame,labels[int(label[0])], (x1+15,y1-10), color= (255,255,255), thickness= 2, fontScale = 1 , fontFace = cv2.FONT_HERSHEY_SIMPLEX)
        cv2.putText(frame, printScore, (x1+35,y1-10), color= (255,255,255), thickness= 2, fontScale = 1 , fontFace = cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow('camera',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()