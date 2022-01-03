import cv2 
import numpy as np
import matplotlib.pyplot as plt 

CAM_PORT = -1
video = "test2.mp4"
if(CAM_PORT == -1):
    cap = cv2.VideoCapture(video)
else:
    cap = cv2.VideoCapture(CAM_PORT, cv2.CAP_DSHOW)

def plt_show(image):
    plt.imshow(image)
    plt.show()

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5),0) 
    return cv2.Canny(blur, 50,150)

def crop(image):
    height = image.shape[0]
    width = image.shape[1]
    if len(image.shape)>2:
        channel_count = image.shape[2]
    else:
        channel_count = 1

    polygon = [
    (0,height),
    (width/2,height/2),
    (width,height)
    ]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([polygon], np.int32),(255,) * channel_count)
    cropped = cv2.bitwise_and(image,mask)

    # for index, point in enumerate(polygon, start = 0):
    #     if(index == len(polygon)-1):
    #         next = 0
    #     else:
    #         next = index+1

    #     cv2.line(cropped, point , polygon[next], (255,)*channel_count, thickness=1)
    return cropped

def hough(canny, orig):
    lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100, minLineLength=10, maxLineGap=25)
    line_image = np.zeros_like(orig)
    if(lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (0,255,0), thickness=5)
    return line_image

def slope_intercept(point1, point2):
    slope = (point1[1]-point2[1])/(point1[0]-point2[0])
    y_intercept = point1[1] - slope*point1[0]
    return (slope, y_intercept)

def avg_slope(lines):
    if(lines is not None):
        pass #???????????????????????????????????????????????????

blank_image = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
win_name = "Kanan"
cv2.imshow(win_name, blank_image)

def window_opened(name):
    try:
        result = cv2.getWindowProperty(name, 0) >= 0
    except:
        result = False
    return result

while cap.isOpened() and window_opened(win_name): #window is opened and capturing initialized
    ret, frame = cap.read()

    canny_frame = canny(frame)
    cropped_frame = crop(canny_frame)
    line_frame = hough(cropped_frame, frame)

    combo = cv2.addWeighted(frame, 1, line_frame, 0.5, 1)
    cv2.imshow(win_name, combo)

    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("p"):
        plt_show(frame)
cap.release()
cv2.destroyAllWindows()