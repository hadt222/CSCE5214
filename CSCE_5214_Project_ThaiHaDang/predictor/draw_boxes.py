from collections import deque
import numpy as np
import math
import cv2
# from .speed_estimator import twolines_speed, twopoints_speed, birdeyes_speed, transform_3d_speed
from speed_estimator import twolines_speed, twopoints_speed, birdeyes_speed, transform_3d_speed

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
speed_queue = {} 

# for twolines method
twolines_queue = {}

# for birdeyes method
transformed_center_deque = {}

object_counter = {}
object_counter1 = {}


# line1 for counting
line1 = [(50, 500), (1200, 500)]
# line2 and line1 for twoline speed
line2 = [(50, 400), (1200, 400)]


# for birdeyes method
# transform_matrix_path = "./predictor/birdeyes/transform_matrix.npy"
transform_matrix_path = "./speed_estimator/birdeyes/transform_matrix.npy"
M = np.load(transform_matrix_path)
M = np.array(M, np.float32)


def xyxy_to_xywh(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str
def draw_boxes(img, bbox, names, speed_method, object_id, identities=None, offset=(0, 0)):
    cv2.line(img, line1[0], line1[1], (46,162,112), 3)
    if speed_method=='twolines':
        cv2.line(img, line2[0], line2[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        # print(box)
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # if using 4points transform
        if speed_method=='birdeyes':
            np_center = np.array(center, dtype=np.float32).reshape(1, -1, 2)
            transformed_center = cv2.perspectiveTransform(np_center, M)
    

        # get ID of object
        id = int(identities[i]) if identities is not None else 0
 
        if id not in data_deque:  
            data_deque[id] = deque(maxlen=64)
            transformed_center_deque[id] = deque(maxlen=64)
            speed_queue[id] = []


        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)

        if speed_method=="twolines" and id in twolines_queue:
            twolines_queue[id] += 1

        # if using 4point transform
        if speed_method=="birdeyes":
            transformed_center_deque[id].appendleft(transformed_center[0][0])
   
        if len(data_deque[id]) >= 2:
            # if using twoline function
            if speed_method=='twolines' and intersect(data_deque[id][0], data_deque[id][1], line1[0], line1[1]) or intersect(data_deque[id][0], data_deque[id][1], line2[0], line2[1]):
                if id not in twolines_queue:
                    twolines_queue[id]=0
                else: 
                    twolines_queue[id]+=1
                    object_speed = twolines_speed(len(twolines_queue), line1=line1, line2=line2)
                    speed_queue[id].append(object_speed)
            # if using twopoint function
            if speed_method=='twopoints':
                object_speed = twopoints_speed(data_deque[id][1], data_deque[id][0])
                speed_queue[id].append(object_speed)

            # if using birdeyes function 
            elif speed_method=='birdeyes':
                object_speed = birdeyes_speed(transformed_center_deque[id][1], transformed_center_deque[id][0])
                speed_queue[id].append(object_speed)

            # if using 3d transform function 
            elif speed_method=='transform_3d':
                object_speed = transform_3d_speed(data_deque[id][1], data_deque[id][0])
                speed_queue[id].append(object_speed)

            # get direction
            direction = get_direction(data_deque[id][0], data_deque[id][1])


            if intersect(data_deque[id][0], data_deque[id][1], line1[0], line1[1]):
                cv2.line(img, line1[0], line1[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1


        try:
            label = label + " " + str(sum(speed_queue[id])//len(speed_queue[id])) + "km/h"
        except:
            pass
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    
    #4. Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" +str(value)
            cv2.line(img, (width - 320,25), (width-20,25), [85,45,255], 40)
            cv2.putText(img, f'Vehicles Entering', (width-20 - 300, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (width - 150, 65 + (idx*40)), (width-20, 65 + (idx*40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.line(img, (20,25), (300,25), [85,45,255], 40)
            cv2.putText(img, f'Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
            cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
            cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
    
    return img