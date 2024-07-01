#라이브러리 설치 
import torch 

#모델 

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
model.conf = 0.91

#이미지 검증 (Predict)
img = "https://www.usatoday.com/gcdn/authoring/authoring-images/2024/02/26/USAT/72746758007-wide-use-am-082423-ive-canon-400-hires.jpg?crop=5009,2819,x0,y260&width=3200&height=1801&format=pjpg&auto=webp"

results = model(img)

#검증
# results.print()
results.pandas().xyxy[0]
results.save()

#crop 
# results.crop()