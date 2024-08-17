import cv2
import numpy as np
import os
import csv
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

video = cv2.VideoCapture(0)
face_detection = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)

with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

resized_faces = [cv2.resize(face, (50, 50)).flatten() for face in FACES]
resized_faces = np.array(resized_faces)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(resized_faces, LABELS)

img_bg = cv2.imread("bg.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        gray_crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile("attendance/attendance_" + date + ".csv")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        attendance = [str(output[0]), str(timestamp)]
    img_bg[162:162+480, 55:55+640] = frame

    cv2.imshow("frame", img_bg)
    k = cv2.waitKey(1)
    if k == ord('o'):
        time.sleep(5)

        if exist:
            with open("attendance/attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("attendance/attendance_" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
            csvfile.close()
        
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()