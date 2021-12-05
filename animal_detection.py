import cv2
import numpy as np
import fileinput as fi

frozenModel = './model/frozen_inference_graph.pb'
configFile  = './model/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
classNames  = []


model = cv2.dnn_DetectionModel(frozenModel, configFile)

model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

videoCapture = cv2.VideoCapture(0)

with fi.input( ('./model/coco.names') ) as f:
    for line in f:
        classNames.append(line.strip('\n'))

print(classNames)

while True:
    ret, frame = videoCapture.read()
    smallFrame = frame # cv2.resize(frame, (0, 0), fx=1.0, fy=1.0)

    classIDs, confidences, bboxArray = model.detect(smallFrame, confThreshold=0.6, nmsThreshold=0.1)

    for classID, confidence, bbox in zip(classIDs, confidences, bboxArray):
        if confidence > 0.4:
            p1 = (bbox[0], bbox[1])
            p2 = (bbox[2], bbox[3])
            name = classNames[classID - 1]
            cv2.rectangle(smallFrame, bbox, color=(255, 0, 0), thickness=2)
            cv2.putText(smallFrame, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(smallFrame, str(confidence), (p1[0], p1[1] + 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 2)
            
            print("Confidence: " + name, confidence) 

    cv2.imshow('Video', smallFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()