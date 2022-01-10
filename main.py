import cv2 
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import append 

CAM_PORT = -1
video = "test2.mp4"
if(CAM_PORT == -1):
    cap = cv2.VideoCapture(video)
else:
    cap = cv2.VideoCapture(CAM_PORT, cv2.CAP_DSHOW)

last_right_slope = 0
last_left_slope =0

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
    return cropped

def hough(canny):
    lines = cv2.HoughLinesP(canny, 2, np.pi/180, 100, minLineLength=10, maxLineGap=25)
    new_lines = []
    if(lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            line = slope_intercept((x1,y1),(x2,y2))
            new_lines.append(line)
    return new_lines


def slope_intercept(point1, point2):
    slope = (point1[1]-point2[1])/(point1[0]-point2[0])
    y_intercept = point1[1] - slope*point1[0]
    return (slope, y_intercept)

def make_points(slope_intercept, image_height):
    try:
        y1 = image_height
        x1 = int((y1 - slope_intercept[1])/slope_intercept[0])
        y2 = int((3/5) * image_height)
        x2 = int((y2 - slope_intercept[1])/slope_intercept[0])
        return((x1,y1), (x2, y2))
    except:
        return None
    

def show_lines(image, lines):
    global last_left_slope, last_right_slope    
    black_copy = np.zeros_like(image)
    
    right_fit = []
    left_fit = []
    offset = 0.3

    if(len(lines)>0):
        for line in lines:
            if(line[0] + offset < 0):
                left_fit.append(line)
            if(line[0] - offset > 0):
                right_fit.append(line)

    if(len(right_fit)>0):
        avg_right = np.average(right_fit, axis=0)
        last_right_slope = avg_right[0]
        right_line = make_points(avg_right, image.shape[0])
        if(right_line!= None):
            cv2.line(black_copy, right_line[0], right_line[1], (0,255,0), 5)
    if(len(left_fit)>0):
        avg_left = np.average(left_fit, axis=0)
        last_left_slope = avg_left[0]
        left_line = make_points(avg_left, image.shape[0])
        if(left_line != None):
            cv2.line(black_copy, left_line[0], left_line[1], (0,255,0), 5)
    
    return black_copy

def show_mask(image, polygon):
    lined_image = np.zeros_like(image)
    for id in range(len(polygon)-1):
        if(id < len(polygon)-1):
            cv2.line(lined_image, polygon[id], polygon[id+1], (255,255,255), thickness=2)
    return cv2.addWeighted(frame, 0.8, lined_image, 0.2, 1)


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
    height = frame.shape[0]
    width = frame.shape[1]

    canny_frame = canny(frame)
    cropped_frame = crop(canny_frame)
    lines = hough(cropped_frame)
    line_frame = show_lines(frame, lines)
    final_frame = cv2.addWeighted(frame, 0.8, line_frame, 0.2, 1)
    #[(0,height), (int(width/2),int(height/2)),(width,height)]
    combo = show_mask(final_frame, [(0,height), (int(width/2),int(height/2)),(width,height)])
    cv2.imshow(win_name, combo)

    if cv2.waitKey(1) == ord("q"):
        break
    if cv2.waitKey(1) == ord("p"):
        plt_show(frame)
cap.release()
cv2.destroyAllWindows()