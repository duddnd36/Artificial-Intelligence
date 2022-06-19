import numpy as np
import cv2
import winsound
classes = []
f = open('./darknet/data/coco.names', 'r')
classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

yolo_model = cv2.dnn.readNet('./darknet/backup/yolov3.weights', './darknet/cfg/yolov3.cfg')  # 욜로 읽어오기
layer_names = yolo_model.getLayerNames()
out_layers = [layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers()]
# YOLO 란: 사진을 격자 형태로 쪼개,

def process_video():  # 비디오에서 침입자 검출해 알리기
    video = cv2.VideoCapture('./darknet/data/fire.mp4')
    while video.isOpened():
        success, img = video.read()
        if success:
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True, crop=False)
            # 오토인코더, 잠재표현
            # • 인공 신경망을 이용하여 입력을 출력의 정답 레이블로 사용하는 모델을
            # 구현할 수 있는데, 이러한 모델을 오토인코더라고 부른다.
            # • 오토인코더의 1단계 학습 모델에서 중간 단계 출력을 생성하고, 이 중간
            # 출력을 입력으로 하는 2단계 학습 모델이 원래의 데이터를 복원하게 하면
            yolo_model.setInput(blob)
            output3 = yolo_model.forward(out_layers)
            # 중간 단계의 출력은 원래 데이터의 중요한 특징을 담게 된다.
            # • 오토인코더의 1단계 모델을 인코더, 2단계 모델을 디코더라고 부른다.
            # • 오토인코더의 중간 계층은 입력 데이터의 잠재 표현을 담게 된다.
            # • 잠재 표현이 존재하는 공간이 잠재 공간이다.
            class_ids, confidences, boxes = [], [], []
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
            # • 잠재 표현의 차원을 조절하여 다양한 크기의 차원 축소가 가능하다.
            # • 오토인코더가 만들어내는 잠재표현을 이용하여 데이터의 압축과 복원이 가능하다.
            # • 오토인코더의 연결망은 일반적인 퍼셉트론 뿐만 아니라 다양한 구조가 가능하므로 데이
            # 터의 특성에 따라 여러 가지 접근법을 사용할 수 있다.
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    text = str(classes[class_ids[i]]) + '%.3f' % confidences[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                    cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[i]], 2)
            # • 이미지 데이터의 압축과 복원을 위해서는 컨볼루션 신경망을 사용하는 오토인코더를 통
            # 해 성능을 높일 수 있다.
            # • 컨볼루션 신경망으로 인코더를 구현하면, 디코더 단계에서는 디컨볼루션을 적용해야 한다.
            # • 컨벌루션 신경망을 이용한 오토인코더는 이미지 데이터의 특징을 잘 유지한 압축과 복원
            # 이 가능하다.

            if 0 in class_ids:  # 사람이 검출됨(0='person')
                winsound.Beep(frequency=2000, duration=500)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break


# 예제 3 비디오와 고나련된 윈도우와 비디오 연결을 종료한다 비디오 연결을 하지 않는다.
# video.release() ,cv2.destroyAllWindows()
process_video()

# • 잠재 표현을 차원 축소에만 사용하지 않고, 입력 데이터가 가진 특징을 파악하는 다양한
# 용도로 사용할 수 있다.
# • 가지고 있는 데이터의 주요 특징을 파악하고, 이러한 특징과 일치하는 새로운 데이터를
# 생성하는 생성 모델도 잠재 표현 개념을 이용하여 고안할 수 있다
