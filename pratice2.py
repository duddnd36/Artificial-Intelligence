import numpy as np
import cv2
import winsound
#05~08행은 물체 부류 이름과 색깔을 설정하는 코드로
classes = []
f = open('./darknet/data/coco.names', 'r')
classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

yolo_model = cv2.dnn.readNet('./darknet/backup/yolov3.weights', './darknet/cfg/yolov3.cfg')  # 10행은 opencv 라이브러인 cv2를 불러온다
layer_names = yolo_model.getLayerNames() #10~12행은 YOLOv3 모델을 읽고 출력층을 알아내는 코드로,
out_layers = [layer_names[i - 1] for i in yolo_model.getUnconnectedOutLayers()]
#14~15행은 웹캠을 통해 비디오 프레임을 처리하는 process_video 함수이다.
def process_video():  # 비디오에서 침입자 검출해 알리기
    video = cv2.VideoCapture('./darknet/data/fire.mp4')   #웹캠과 연결하고 웹캠 정보를 video객체에 저장한다.
    while video.isOpened(): #16~55행은 입력되는 비디오를 검사하다가 키보드에서 esc에 해당하는 문자 27이 들어오면 루프를 탈출한다.
        success, img = video.read()
        if success:
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 1.0 / 256, (448, 448), (0, 0, 0), swapRB=True, crop=False)

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

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    text = str(classes[class_ids[i]]) + '%.3f' % confidences[i]
                    cv2.rectangle(img, (x, y), (x + w, y + h), colors[class_ids[i]], 2)
                    cv2.putText(img, text, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, colors[class_ids[i]], 2)

            cv2.imshow('Object detection', img)
#50~21행은 검출된 물체 목록, 즉 class_ids 객체에 사람이 포함되면 침입자가 나타난 것으로간주하고, “사람이 나타나다!!!”라는 메시지와 함게 경고음을 발생한다
            if 0 in class_ids:  # 사람이 검출됨(0='person')
                print('사람이 나타났다!!!')
                winsound.Beep(frequency=2000, duration=500)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: break
#57행은 비디오 연결을 종료하고, 58행은 비디오와 관련된 윈도우를 끈다,
    video.release()
    cv2.destroyAllWindows()


process_video()