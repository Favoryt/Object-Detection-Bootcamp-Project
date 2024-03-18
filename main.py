from  ultralytics import  YOLO
import cv2
# import numpy as np

model = YOLO("yolov8n.pt")
image = cv2.imread("people-walking-on-road.jpg")

result = model(image, save = True, conf = .6)


for i,r in enumerate(result):
    detection = r.boxes.data.tolist()

    names = r.names
    classes = r.boxes.cls.tolist()

    for labels,detection in zip(classes,detection):

        label = names[labels]
        x,y,w,h,conf,_ = detection

        # print(conf)


        cv2.putText(image, str(label), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(conf), (int(w), int(h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 100), 1, cv2.LINE_AA)

        cv2.rectangle(image,(int(x),int(y)),(int(w),int(h)), (0,255,0),2)

    cv2.imshow("image",image)

    cv2.waitKey(0)