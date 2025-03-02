from utils import *
'''def hand_gesture():
    import cv2
    import numpy as np
    import pyautogui
    import time
    import pickle
    import sys
    print('**********************************************************************')
    print(' For new users run callibrate module first')
    print(' Ignore if already done!!')
    print('**********************************************************************')
    with open("range.pickle","rb") as file:
        a = pickle.load(file)
        try:
            red_range1 = np.array([a['hsv_1']['lower_bound'],a['hsv_1']['upper_bound']])
            blue_range1 = np.array([a['hsv_2']['lower_bound'],a['hsv_2']['upper_bound']])
            yellow_range1 = np.array([a['hsv_3']['lower_bound'],a['hsv_3']['upper_bound']])
        except:
            print("Callibration required")
            import callibrate
    global yellow_range,red_range,blue_range
    blue_range = np.array([[88,78,20],[128,255,255]])
    yellow_range = np.array([[65,60,60],[80,255,255]])
    red_range = np.array([[150,85,72],[180 ,255,255]])
    print(yellow_range)
    b_cen, y_pos, r_cen = [240,320],[240,320],[240,320]
    cursor = [960,540]
    r_area = [100,1700]
    b_area = [100,1700]
    y_area = [100,1700]
    global kernel
    kernel = np.ones((7,7),np.uint8)
    global perform
    perform = False
    global showCentroid
    showCentroid = False
    def nothing(x):
        pass
    def swap( array, i, j):
        temp = array[i]
        array[i] = array[j]
        array[j] = temp
    def distance( c1, c2):
        distance = pow( pow(c1[0]-c2[0],2) + pow(c1[1]-c2[1],2) , 0.5)
        return distance
    def changeStatus(key):
        global perform
        global showCentroid
        global yellow_range,red_range,blue_range
        if key == ord('p'):
            perform = not perform
            if perform:
                print('Mouse simulation ON...')
            else:
                print('Mouse simulation OFF...')
        elif key == ord('c'):
            showCentroid = not showCentroid
            if showCentroid:
                print('Showing Centroids...')
            else:
                print('Not Showing Centroids...')
        elif key == ord('r'):
            print('**********************************************************************')
            print(' You have entered refining mode.')
            print(' Use the trackbars to refine segmented colors and press SPACE when done.')
            print(' Press D to use the default settings.')
            print('**********************************************************************')
            yellow_range = calibrateColor('Yellow', yellow_range)
            red_range = calibrateColor('Red', red_range)
            blue_range = calibrateColor('Blue', blue_range)
        else:
            pass
    def makeMask(hsv_frame, color_Range):
        mask = cv2.inRange( hsv_frame, color_Range[0], color_Range[1])
        eroded = cv2.erode( mask, kernel, iterations=1)
        dilated = cv2.dilate( eroded, kernel, iterations=1)
        return dilated
    def drawCentroid(vid, color_area, mask, showCentroid):
        contour,hrchy = cv2.findContours( mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        l=len(contour)
        area = np.zeros(l)
        for i in range(l):
            if cv2.contourArea(contour[i])>color_area[0] and cv2.contourArea(contour[i])<color_area[1]:
                area[i] = cv2.contourArea(contour[i])
            else:
                area[i] = 0
        a = sorted( area, reverse=True)
        for i in range(l):
            for j in range(1):
                if area[i] == a[j]:
                    swap( contour, i, j)
        if l > 0 :
            M = cv2.moments(contour[0])
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                center = (cx,cy)
                if showCentroid:
                    cv2.circle( vid, center, 5, (0,0,255), -1)
                return center
        else:
            return (-1,-1)
    def calibrateColor(color, def_range):
        global kernel
        name = 'Refine '+ color
        cv2.namedWindow(name)
        cv2.createTrackbar('Hue', name, def_range[0][0]+20 , 180, nothing)
        cv2.createTrackbar('Sat', name, def_range[0][1] , 255, nothing)
        cv2.createTrackbar('Val', name, def_range[0][2] , 255, nothing)
        while(1):
            ret , frameinv = cap.read()
            frame=cv2.flip(frameinv ,1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hue = cv2.getTrackbarPos('Hue', name)
            sat = cv2.getTrackbarPos('Sat', name)
            val = cv2.getTrackbarPos('Val', name)
            lower = np.array([hue-20,sat,val])
            upper = np.array([hue+20,255,255])
            mask = cv2.inRange(hsv, lower, upper)
            eroded = cv2.erode( mask, kernel, iterations=1)
            dilated = cv2.dilate( eroded, kernel, iterations=1)
            cv2.imshow(name, dilated)
            k = cv2.waitKey(5) & 0xFF
            if k == ord(' '):
                cv2.destroyWindow(name)
                return np.array([[hue-20,sat,val],[hue+20,255,255]])
            elif k == ord('d'):
                cv2.destroyWindow(name)
                return def_range
    def setCursorPos( yc, pyp):
        yp = np.zeros(2)
        if abs(yc[0]-pyp[0])<5 and abs(yc[1]-pyp[1])<5:
            yp[0] = yc[0] + .7*(pyp[0]-yc[0])
            yp[1] = yc[1] + .7*(pyp[1]-yc[1])
        else:
            yp[0] = yc[0] + .1*(pyp[0]-yc[0])
            yp[1] = yc[1] + .1*(pyp[1]-yc[1])
        return yp
    def chooseAction(yp, rc, bc):
        out = np.array(['move', 'false'])
        if rc[0] != -1 and bc[0] != -1:
            if distance(yp, rc) < 50 and distance(yp, bc) < 50 and distance(rc, bc) < 50:
                out[0] = 'drag'
                out[1] = 'true'
                return out
            elif distance(rc, bc) < 40:
                out[0] = 'left'
                return out
            elif distance(yp, rc) < 40:
                out[0] = 'right'
                return out
            elif distance(yp, rc) > 40 and rc[1] - bc[1] > 120:
                out[0] = 'down'
                return out
            elif bc[1] - rc[1] > 110:
                out[0] = 'up'
                return out
            else :
                return out
        else :
            out[0] = -1
            return out
    def performAction(yp, rc, bc, action, drag, perform):
        if perform:
            cursor[0] = 4 * (yp[0] - 110)
            cursor[1] = 4 * (yp[1] - 120)
            if action == 'move':
                if yp[0] > 110 and yp[0] < 590 and yp[1] > 120 and yp[1] < 390:
                    pyautogui.moveTo(cursor[0], cursor[1])
                elif yp[0] < 110 and yp[1] > 120 and yp[1] < 390:
                    pyautogui.moveTo(8, cursor[1])
                elif yp[0] > 590 and yp[1] > 120 and yp[1] < 390:
                    pyautogui.moveTo(1912, cursor[1])
                elif yp[0] > 110 and yp[0] < 590 and yp[1] < 120:
                    pyautogui.moveTo(cursor[0], 8)
                elif yp[0] > 110 and yp[0] < 590 and yp[1] > 390:
                    pyautogui.moveTo(cursor[0], 1072)
                elif yp[0] < 110 and yp[1] < 120:
                    pyautogui.moveTo(8, 8)
                elif yp[0] < 110 and yp[1] > 390:
                    pyautogui.moveTo(8, 1072)
                elif yp[0] > 590 and yp[1] > 390:
                    pyautogui.moveTo(1912, 1072)
                else :
                    pyautogui.moveTo(1912, 8)
            elif action == 'left':
                pyautogui.click(button = 'left')
            elif action == 'right':
                pyautogui.click(button = 'right')
                time.sleep(0.3)
            elif action == 'up':
                pyautogui.scroll(5)
            elif action == 'down':
                pyautogui.scroll(-5)
            elif action == 'drag' and drag == 'true':
                global y_pos
                drag = 'false'
                pyautogui.mouseDown()
                while (1):
                    k = cv2.waitKey(10) & 0xFF
                    changeStatus(k)
                    _, frameinv = cap.read()
                    frame = cv2.flip(frameinv, 1)
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    b_mask = makeMask(hsv, blue_range1)
                    r_mask = makeMask(hsv, red_range1)
                    y_mask = makeMask(hsv, yellow_range1)
                    py_pos = y_pos
                    b_cen = drawCentroid(frame, b_area, b_mask, showCentroid)
                    r_cen = drawCentroid(frame, r_area, r_mask, showCentroid)
                    y_cen = drawCentroid(frame, y_area, y_mask, showCentroid)
                    if py_pos[0] != -1 and y_cen[0] != -1:
                        y_pos = setCursorPos(y_cen, py_pos)
                    performAction(y_pos, r_cen, b_cen, 'move', drag, perform)
                    cv2.imshow('Frame', frame)
                    if distance(y_pos, r_cen) > 60 or distance(y_pos, b_cen) > 60 or distance(r_cen, b_cen) > 60:
                        break
                pyautogui.mouseUp()
    cap = cv2.VideoCapture(0)
    print('**********************************************************************')
    print(' You have entered refining mode.')
    print(' Use the trackbars to refine segmented colors and press SPACE when done.')
    print(' Press D to use the default settings.')
    print('**********************************************************************')
    yellow_range = calibrateColor('Yellow', yellow_range)
    red_range = calibrateColor('Red', red_range)
    blue_range = calibrateColor('Green', blue_range)
    print(' Refining Successfull...')
    cv2.namedWindow('Frame')
    print('**********************************************************************')
    print(' Press P to turn ON and OFF mouse simulation.')
    print(' Press C to display the centroid of various colours.')
    print(' Press R to refine color ranges.')
    print(' Press ESC to exit.')
    print('**********************************************************************')
    pyautogui.FAILSAFE = False
    while (1):
        k = cv2.waitKey(10) & 0xFF
        changeStatus(k)
        _, frameinv = cap.read()
        frame = cv2.flip(frameinv, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        b_mask = makeMask(hsv, blue_range)
        r_mask = makeMask(hsv, red_range)
        y_mask = makeMask(hsv, yellow_range)
        py_pos = y_pos
        b_cen = drawCentroid(frame, b_area, b_mask, showCentroid)
        r_cen = drawCentroid(frame, r_area, r_mask, showCentroid)
        y_cen = drawCentroid(frame, y_area, y_mask, showCentroid)
        if py_pos[0] != -1 and y_cen[0] != -1 and y_pos[0] != -1:
            y_pos = setCursorPos(y_cen, py_pos)
        output = chooseAction(y_pos, r_cen, b_cen)
        if output[0] != -1:
            performAction(y_pos, r_cen, b_cen, output[0], output[1], perform)
        cv2.imshow('Frame', frame)
        if k == 27:
            print("Quitting")
        break
    cap.release()
    cv2.destroyAllWindows()'''
def facial_gesture():
    from imutils import face_utils
    import numpy as np
    import pyautogui as pyag
    import imutils
    import dlib
    import cv2
    MOUTH_AR_THRESH = 0.6
    MOUTH_AR_CONSECUTIVE_FRAMES = 15
    EYE_AR_THRESH = 0.25
    EYE_AR_CONSECUTIVE_FRAMES = 15
    WINK_AR_DIFF_THRESH = 0.04
    WINK_AR_CLOSE_THRESH = 0.19
    WINK_CONSECUTIVE_FRAMES = 10
    MOUTH_COUNTER = 0
    EYE_COUNTER = 0
    WINK_COUNTER = 0
    INPUT_MODE = False
    EYE_CLICK = False
    LEFT_WINK = False
    RIGHT_WINK = False
    SCROLL_MODE = False
    ANCHOR_POINT = (0, 0)
    WHITE_COLOR = (255, 255, 255)
    YELLOW_COLOR = (0, 255, 255)
    RED_COLOR = (0, 0, 255)
    GREEN_COLOR = (0, 255, 0)
    BLUE_COLOR = (255, 0, 0)
    BLACK_COLOR = (0, 0, 0)
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
        if diff_ear > WINK_AR_DIFF_THRESH:
            if leftEAR < rightEAR:
                if leftEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1
                    if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                        pyag.click(button = 'left')
                        cv2.putText(frame, 'Left Click!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,0.7, RED_COLOR, 2)
                        WINK_COUNTER = 0
            elif leftEAR > rightEAR:
                if rightEAR < EYE_AR_THRESH:
                    WINK_COUNTER += 1
                    if WINK_COUNTER > WINK_CONSECUTIVE_FRAMES:
                        pyag.click(button = 'right')
                        cv2.putText(frame, 'Right Click!', (10, 60), cv2.FONT_HERSHEY_SIMPLEX,0.7, RED_COLOR, 2)
                        WINK_COUNTER = 0
            else :
                WINK_COUNTER = 0
        else :
            if ear <= EYE_AR_THRESH:
                EYE_COUNTER += 1
                if EYE_COUNTER > EYE_AR_CONSECUTIVE_FRAMES:
                    SCROLL_MODE = not SCROLL_MODE
                    EYE_COUNTER = 0
            else :
                EYE_COUNTER = 0
                WINK_COUNTER = 0
        if mar > MOUTH_AR_THRESH:
            MOUTH_COUNTER += 1
            if MOUTH_COUNTER >= MOUTH_AR_CONSECUTIVE_FRAMES:
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
            if dir == 'right':
                pyag.moveRel(drag, 0)
            elif dir == 'left':
                pyag.moveRel(-drag, 0)
            elif dir == 'up':
                if SCROLL_MODE:
                    pyag.scroll(40)
                else :
                    pyag.moveRel(0, -drag)
            elif dir == 'down':
                if SCROLL_MODE:
                    pyag.scroll(-40)
                else :
                    pyag.moveRel(0, drag)
        if SCROLL_MODE:
            cv2.putText(frame, 'SCROLL MODE IS ON!', (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED_COLOR, 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    cv2.destroyAllWindows()
    vid.release()
facial_gesture()