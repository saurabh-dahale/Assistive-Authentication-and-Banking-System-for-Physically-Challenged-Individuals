import cv2
img = cv2.imread("test_image.png")
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(hsv,cv2.COLOR_BGR2HSV)
cv2.imshow("image",hsv2)