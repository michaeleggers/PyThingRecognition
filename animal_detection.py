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

    classIDs, confidence, bboxArray = model.detect(smallFrame, confThreshold=0.6, nmsThreshold=0.2)

    # print("length classIDs: ", len(classIDs))
    # print("length bbox: ", len(bbox))
    # print(len(classIDs) == len(bbox))
    for i, classID in enumerate(classIDs):
        # p2 = (bbox[0], bbox[1])
        # print(bbox)
        bbox = bboxArray[i]
        p1 = (bbox[0], bbox[1])
        p2 = (bbox[2], bbox[3])
        name = classNames[classID - 1]
        cv2.rectangle(smallFrame, bbox, color=(255, 0, 0), thickness=2)
        cv2.putText(smallFrame, name, p1, cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
        print("Confidence: " + name, confidence[i])
        # if classID == 19:
        #     cv2.putText(smallFrame, "Horse!", (200, 200), 1, 3.0, (244, 244, 244))
        # elif classID == 20:
        #     cv2.putText(smallFrame, "Sheep!", (200, 200), 1, 3.0, (244, 244, 244))

            # for box in bbox:
            #     p1 = (box[2], box[3])
            #     p2 = (box[0], box[1])
            #     cv2.rectangle(smallFrame, p1, p2, (0, 255, 255))

    cv2.imshow('Video', smallFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()