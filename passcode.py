from utils import *
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
MOUTH_AR_THRESH = 0.6
MOUTH_AR_CONSECUTIVE_FRAMES = 15
MOUTH_COUNTER = 0
INPUT_MODE = False
ANCHOR_POINT = (0, 0)
YELLOW_COLOR = (0, 255, 255)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
shape_predictor = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
vid = cv2.VideoCapture(0)
resolution_w = 1366
resolution_h = 768
cam_w = 640
cam_h = 480
unit_w = resolution_w / cam_w
unit_h = resolution_h / cam_h
inp1=0
inp2=0
password = [1,3,4,4,2]
k = []
while True:
    _, frame = vid.read()
    frame = imutils.resize(frame, width = cam_w, height = cam_h)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        rect = rects[0]
    else :
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        continue
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    mouth = shape[mStart: mEnd]
    leftEye = shape[lStart: lEnd]
    rightEye = shape[rStart: rEnd]
    nose = shape[nStart: nEnd]
    temp = leftEye
    leftEye = rightEye
    rightEye = temp
    mar = mouth_aspect_ratio(mouth)
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
    diff_ear = np.abs(leftEAR - rightEAR)
    nose_point = (nose[3, 0], nose[3, 1])
    mouthHull = cv2.convexHull(mouth)
    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [mouthHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [leftEyeHull], -1, YELLOW_COLOR, 1)
    cv2.drawContours(frame, [rightEyeHull], -1, YELLOW_COLOR, 1)
    for (x, y) in np.concatenate((mouth, leftEye, rightEye), axis = 0):
        cv2.circle(frame, (x, y), 2, GREEN_COLOR, -1)
    if mar > MOUTH_AR_THRESH:
        MOUTH_COUNTER += 1
        if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES and inp2==0:
            print("Enter Password")
            inp2=1
            INPUT_MODE = not INPUT_MODE
            MOUTH_COUNTER = 0
            ANCHOR_POINT = nose_point
        elif MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES and inp2==1:
            if k==password:
                print("Authentication Successful")
                k=[]
            else:
                print("Password Incorrect")
                k=[]
            inp2=0
            INPUT_MODE = not INPUT_MODE
            MOUTH_COUNTER = 0
            ANCHOR_POINT = nose_point
    else :
        MOUTH_COUNTER = 0
    if INPUT_MODE:
        cv2.putText(frame, "READING INPUT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, RED_COLOR, 2)
        x, y = ANCHOR_POINT
        nx, ny = nose_point
        w, h = 60, 35
        multiple = 1
        cv2.rectangle(frame, (x - w, y - h), (x + w, y + h), GREEN_COLOR, 2)
        cv2.line(frame, ANCHOR_POINT, nose_point, BLUE_COLOR, 2)
        dir = direction(nose_point, ANCHOR_POINT, w, h)
        cv2.putText(frame, dir.upper(), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7,RED_COLOR, 2)
        drag = 18
        if dir == 'right' and inp1==0:
            print("Right")
            inp1=1
            k.append(4)
        elif dir == 'left' and inp1==0:
            print("Left")
            inp1=1
            k.append(3)
        elif dir == 'up' and inp1==0:
            print("Up")
            inp1=1
            k.append(1)
        elif dir == 'down' and inp1==0:
            print("Down")
            inp1=1
            k.append(2)
        elif dir =='none' and inp1==1:
            inp1=0
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
cv2.destroyAllWindows()
vid.release()