# https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
# https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
"""
    This code for detect face from video
    The process goes:
        1) load the video
        2) capture the one frame from the video
        3) get the area of detected faces in the frame
        4) visualize or crop the face area

    @author Joonho Wohn
    @since  18-04-01

"""


import cv2 as cv
import numpy as np
import os
import datetime

VIDEO_NAME = "reallyreally.mp4"
now = str(datetime.datetime.now())[:19].replace(' ', '-').replace(':', '-')
DIRECTORY_NAME = "Cropped_imgs/" + now + '-' + VIDEO_NAME.split('.')[0]

def make_directory():
    if not os.path.exists(DIRECTORY_NAME):
        os.makedirs(DIRECTORY_NAME)

# this function return frame by play_speed
def get_frame(video_capture, play_speed=1):
    curr_frame = video_capture.get(cv.CAP_PROP_POS_FRAMES)
    video_capture.set(cv.CAP_PROP_POS_FRAMES, curr_frame + play_speed)
    ret, img = video_capture.read()
    fps = video_capture.get(cv.CAP_PROP_FPS)
    return img, (curr_frame + play_speed) / fps

# this function return detect or unable to detect ,rectangle marked image
def get_area_of_frame_face_recognition(img, face_cascade):
    grayed_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # def detectMultiScale(self, image, scaleFactor=None, minNeighbors=None, flags=None, minSize=None, maxSize=None)
    face_area = face_cascade.detectMultiScale(image=grayed_img,scaleFactor=1.3,minNeighbors=5)
    return face_area

# this function show image and quit when press q or end
def show_img(img, faces):
    if len(faces) != 0:
        for (x, y, w, h) in faces:
            c_x = (x + w) / 2
            c_y = (y + h) / 2
            l = min(w, h)
            cv.rectangle(img, (x, y), (x + l, y + l), (255, 0, 0), 1)
    cv.imshow('hello', img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        return False
    else:
        return True

def save_cropped_img(img, faces, sec):
    if len(faces) != 0:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        i = 0
        for (x, y, w, h) in faces:
            print(DIRECTORY_NAME + "/" + str(sec).replace('.','_') + "_" + str(i) + ".jpg")
            face_img =  img[y:y + h, x: x + w]
            # shrink = cv.resize(face_img, None, fx=28/w, fy=28/h, interpolation=cv.INTER_AREA)
            shrink = cv.resize(face_img, (28, 28), interpolation=cv.INTER_AREA)
            # cv.imshow('original', img[c_y:c_y + l, c_x: c_x + l])
            cv.imshow('hello', shrink)
            cv.imwrite(DIRECTORY_NAME + "/" + str(sec).replace('.','_') + "_" + str(i) + ".jpg", shrink)
            i += 1
    if cv.waitKey(1) & 0xFF == ord('q'):
        return False
    else:
        return True

if __name__ == "__main__":
    make_directory()
    # load video
    m_video_capture = cv.VideoCapture(VIDEO_NAME)
    m_face_cascade = cv.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    TOTAL_FRAME = m_video_capture.get(cv.CAP_PROP_FRAME_COUNT)

    while (m_video_capture.get(cv.CAP_PROP_POS_FRAMES) < TOTAL_FRAME):
        img, sec = get_frame(video_capture=m_video_capture, play_speed=10)
        try:
            faces_area = get_area_of_frame_face_recognition(img=img, face_cascade=m_face_cascade)
        except:
            break
        if len(faces_area) != 0:
            print(sec)

        # keyboard interrupt (q)
        # if not show_img(img, faces_area):
        #     break
        if not save_cropped_img(img, faces_area, sec):
            break

