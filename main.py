import cv2 
import numpy

CAM_PORT = 0

cam = cv2.VideoCapture(CAM_PORT, cv2.CAP_DSHOW)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0) 
    return blur

while cam.isOpened():
    ret, frame = cam.read()

    cv2.imshow("Kanan", canny(frame))

    if cv2.waitKey(1) == ord("q"):
        break

#cam.release()
cv2.destroyAllWindows()