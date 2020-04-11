# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:04:30 2020

@author: HP
"""

import cv2
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade=cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
while cap.isOpened():
    _,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.1,4)
    count=0
    for (x,y,h,w) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray=gray[x:x+w,y:y+h]
        roi_color=img[x:x+w,y:y+h]
        eyes=eyes_cascade.detectMultiScale(roi_gray)
        for (ex,ey,eh,ew) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),5)
        count=count+1
    cv2.imshow('Frame',img)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()
print(count)
