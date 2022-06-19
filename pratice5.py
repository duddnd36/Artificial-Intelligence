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
# 자료형태 : 범주형 :명목형, 순서형	 수치형 : 이산형 연속형
# 사이킷런 : 데이터 분석 및 머신러닝 적용을 위한 파이썬 기반 라이브러리
# 패턴(pattern)이란 일정한 특징, 양식, 유형, 틀 등을 말함
# •패턴인식은 패턴이나 특징적 경향을 발견하여 인식하는 것
def process_video():  # 비디오에서 침입자 검출해 알리기
    video = cv2.VideoCapture('./darknet/data/fire.mp4')
    while video.isOpened():
        success, img = video.read()
        # 전통적인 패턴인식: 사전의 정보를 컴퓨터에 기억
        # 인공지능 에서의 패턴인식: 인간의 학습승력과 추론능력을 모델링
        # ex 신경망 과 딥러닝 이후 학습을 통한 패턴인식
        if success:
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True, crop=False)
            # 음성인식: 은닉 마르코프 방법과 딥러닝 방법이 많이 쓰임
            #              (화자 종속(특성인의 음성만 인식),화자 독립 음성인식)
            yolo_model.setInput(blob)
            output3 = yolo_model.forward(out_layers)

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
            # 영상인식: 순환 신경망의 딥러닝으로 음성을 인식하는 시스템
            # ex: 여러 개의 얼굴과 물체 인식, 딥러닝
            # 패턴 인식 기술과 관계 : 감정인식
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            # GAN이란,: 가짜 데이터와 진짜 데이터를 구분하는 신경망
            #Generator : 위조 데이터 생성,Discriminator:위조데이터 적발
            #           점점 더 정교하게 위조            위조데이터 구분 솜씨 향항
            #           진본과 구분할수없을 정도          더이상 구분 불가
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    text = str(classes[class_ids[i]]) + '%.3f' % confidences[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                    cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[i]], 2)


            if 0 in class_ids:  # 사람이 검출됨(0='person')
                winsound.Beep(frequency=2000, duration=500)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break


# 예제 3 비디오와 고나련된 윈도우와 비디오 연결을 종료한다 비디오 연결을 하지 않는다.
# video.release() ,cv2.destroyAllWindows()
process_video()


