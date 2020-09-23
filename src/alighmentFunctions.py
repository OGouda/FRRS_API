import dlib
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import face_recognition

def firstMethod(imgPIL):
    imgArray = np.array(imgPIL)     
    face_locations = face_recognition.face_locations(imgArray)
    face_encodings=[]
    if len(face_locations)>1:
        big_area = 0
        for top, right, bottom, left in face_locations:
            width = right - left
            height = bottom - top
            area = width * height
            if area > big_area:
                big_area = area
                big_location = top,right,bottom,left
        face_locations = [big_location]  

        face_encodings = face_recognition.face_encodings(imgArray, face_locations) 
        IDX_format = 1

    if len(face_locations):
        face_encodings = face_recognition.face_encodings(imgArray, face_locations) 
        return 1, face_encodings

#This method will apply:: Alignment  + cropping
def secondMethod(imgPIL):    
    rotated_img =alignment_dlib(imgPIL)                         #output is numpy
    stacked_img = np.stack((rotated_img,)*3, axis=-1)
    face_encodings = face_recognition.face_encodings(stacked_img) 

    IDX_format = 2
    if len(face_encodings):
        #db_ourImages.append(IDX_format, face_encodings)
        return 2, face_encodings

#This method will apply:: Brightness  +  Alignment + cropping 
def thirdMethod(imgPIL):
    enhancer = ImageEnhance.Brightness(imgPIL)  # Image brightness enhancer
    factor = 1.5                                # Brightens the image by this factor
    im_output = enhancer.enhance(factor)
    rotated_img =alignment_dlib(im_output)      # Align the image
    stacked_img = np.stack((rotated_img,)*3, axis=-1)
    face_encodings = face_recognition.face_encodings(stacked_img)  
    IDX_format = 3
    if len(face_encodings):              
        #db_ourImages.append(IDX_format, face_encoding)
        return 3, face_encodings


def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def cosine_formula(length_line1, length_line2, length_line3):
    cos_a = -(length_line3 ** 2 - length_line2 ** 2 - length_line1 ** 2) / (2 * length_line2 * length_line1)
    return cos_a

def rotate_point(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def shape_to_normal(shape):
    shape_normal = []
    for i in range(0, 5):
        shape_normal.append((i, (shape.part(i).x, shape.part(i).y)))
    return shape_normal

def is_between(point1, point2, point3, extra_point):
    c1 = (point2[0] - point1[0]) * (extra_point[1] - point1[1]) - (point2[1] - point1[1]) * (extra_point[0] - point1[0])
    c2 = (point3[0] - point2[0]) * (extra_point[1] - point2[1]) - (point3[1] - point2[1]) * (extra_point[0] - point2[0])
    c3 = (point1[0] - point3[0]) * (extra_point[1] - point3[1]) - (point1[1] - point3[1]) * (extra_point[0] - point3[0])
    if (c1 < 0 and c2 < 0 and c3 < 0) or (c1 > 0 and c2 > 0 and c3 > 0):
        return True
    else:
        return False

def get_eyes_nose_dlib(shape):
    nose = shape[4][1]
    left_eye_x = int(shape[3][1][0] + shape[2][1][0]) // 2
    left_eye_y = int(shape[3][1][1] + shape[2][1][1]) // 2
    right_eyes_x = int(shape[1][1][0] + shape[0][1][0]) // 2
    right_eyes_y = int(shape[1][1][1] + shape[0][1][1]) // 2
    return nose, (left_eye_x, left_eye_y), (right_eyes_x, right_eyes_y)

def get_eyes_nose(eyes, nose):
    left_eye_x = int(eyes[0][0] + eyes[0][2] / 2)
    left_eye_y = int(eyes[0][1] + eyes[0][3] / 2)
    right_eye_x = int(eyes[1][0] + eyes[1][2] / 2)
    right_eye_y = int(eyes[1][1] + eyes[1][3] / 2)
    nose_x = int(nose[0][0] + nose[0][2] / 2)
    nose_y = int(nose[0][1] + nose[0][3] / 2)
    return (nose_x, nose_y), (right_eye_x, right_eye_y), (left_eye_x, left_eye_y)



def alignment_dlib(imgPIL, test=False):
    RotationThreshold = 10
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('models/shape_predictor_5_face_landmarks.dat') 
    
    ImgPIL_grayed = imgPIL.convert("L")
    gray=np.array(ImgPIL_grayed)

    rects = detector(gray, 0)
    if len(rects) < 1:
        if test:print('No faces! hand it to face_rec.')
        return gray
    else:    
        if test:print('Number of rects: ', len(rects))

        big_area = 0
        big_rect = None
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()

            area = int(w-x) * int(h-y)
            if area > big_area:
                big_area = area
                big_rect = rect   

        x = big_rect.left()
        y = big_rect.top()
        w = big_rect.right()
        h = big_rect.bottom()
        
        #Start preparing the rotation
        shape = predictor(gray, big_rect)
        shape = shape_to_normal(shape)
        nose, left_eye, right_eye = get_eyes_nose_dlib(shape)
        center_of_forehead = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        center_pred = (int((x + w) / 2), int((y + y) / 2))
        length_line1 = distance(center_of_forehead, nose)
        length_line2 = distance(center_pred, nose)
        length_line3 = distance(center_pred, center_of_forehead)
        cos_a = cosine_formula(length_line1, length_line2, length_line3)
        angle = np.arccos(cos_a)
                
        if not np.isnan(angle):
            rotated_point = rotate_point(nose, center_of_forehead, angle)
            rotated_point = (int(rotated_point[0]), int(rotated_point[1]))

            if is_between(nose, center_of_forehead, center_pred, rotated_point):
                angle = np.degrees(-angle)
            else:
                angle = np.degrees(angle)

            if abs(angle) > RotationThreshold: #dont allow less than 10degree
                ImgPIL_grayed = ImgPIL_grayed.rotate(angle)   

                if test:
                    ImgPIL_grayed.save("drawingxx.jpg")

                gray=np.array(ImgPIL_grayed) 
                
    if test:print('Code to crop', gray.shape)

    rects = detector(gray, 0)
    if test:print('To crop. Number of rects:', len(rects))
    if len(rects) > 0:
        big_area = 0         
        for rect in rects:
            x = rect.left()
            y = rect.top()
            w = rect.right()
            h = rect.bottom()

            area = int(w-x) * int(h-y)
            if area > big_area:
                big_area = area
                big_rect = rect 

            x = big_rect.left()
            y = big_rect.top()
            w = big_rect.right()
            h = big_rect.bottom()

        if test:
            draw = ImageDraw.Draw(ImgPIL_grayed)  #
            xy= (x,y,w,h)
            draw.rectangle(xy, fill = None, outline=255)
            ImgPIL_grayed.save("drawingxxx.jpg")
            print('y,h,x,w', y,h,x,w)
            
        if x <0: x = 0
        if y <0: y = 0            
        if h <0: h = 0 
        if w <0: w = 0 

        crop_img = gray[y:h, x:w]
        return crop_img
    
    else:
        print('No faces! hand it to face_rec.')
        return gray

