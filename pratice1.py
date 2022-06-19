import numpy as np
import cv2

classes = []  # 04-06행은 80개의 부류 이름을 파일에서 읽는 코딩이다
f = open('./darknet/data/coco.names', 'r')
classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # 80개 부류를 구분하기 위해 난수를 서로 다른 색깔을 설정한다.

img = cv2.imread('./darknet/data/dog.jpg') #• 09-11행은 테스트영상을 읽고 blobFromImage 함수를 이용해 전처리를 수행한다
height, width, channels = img.shape
blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True, crop=False)

yolo_model = cv2.dnn.readNet('./darknet/backup/yolov3.weights', './darknet/cfg/yolov3.cfg') #13행은 YOLO3 모델을 불러온다
layer_names = yolo_model.getLayerNames() #신경망의 출력을 담당하는 3개의 층을 나타낸다
out_layers = [layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers()]

yolo_model.setInput(blob) #17행은 신경망에 테스트영상을 입력하고
output3 = yolo_model.forward(out_layers) #18행은 출력을 받아 output3 객체에 저장한다.

class_ids, confidences, boxes = [], [], [] #20~32행은 최고 부류 확률이 0.5를 넘는 바운딩 박스 정보를 모은다.
for output in output3:
    for vec85 in output:
        scores = vec85[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # 신뢰도가 50% 이상인 경우만 취함
            centerx, centery = int(vec85[0] * width), int(vec85[1] * height)  # [0,1] 표현을 영상 크기로 변환
            w, h = int(vec85[2] * width), int(vec85[3] * height)
            x, y = int(centerx - w / 2), int(centery - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #비최대 억제알고리즘을 적용해 주위 바운딩 박스에 비해 최대를 유지한 것만 남긴다.
#36~ 41행 비최대 억제에서 살아남은 바운딩 박스를 부류이름, 부류 확률 정보와 함게영상에 표시한다.
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        text = str(classes[class_ids[i]]) + '%.3f' % confidences[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
        cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[i]], 2)
#43행은 새로운 창에 영상을 표시하며
cv2.imshow("Object detection", img)
cv2.waitKey(0) #44행은 키보드의 키가 눌러질때까지 창을 유지한다.
cv2.destroyAllWindows() #45행은 모든 창을 닫고 프로그램을 종료한다