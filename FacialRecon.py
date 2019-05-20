import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as numP
from decimal import Decimal
from PIL import Image
import glob
import xlrd
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from numpy.linalg import inv
import gc
import random
import math
from decimal import Decimal
##open source imported as facial model
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)
numP.seterr(all='ignore')
flat_arr_images_X_matrix=[]#all images in a folder for trainning X
flat_arr_images_Y1_matrix=[]#all images in a folder for trainning predict Y
n=0#iteration of scans
tData = pd.read_excel(r"train.xlsx", sheet_name='Sheet1')
print("Column headings:", tData.columns)
## Trainning model set up
for filename in glob.glob(r"train\*.PNG"):
    im=Image.open(filename).resize((64,64)).convert('RGBA')#resize to smallest due memroy and matrix issue bestfit 64，64
    print(filename)
    flat_arr_images_X_matrix.append(numP.array(im).ravel())#  here we can add bias, 
    flat_arr_images_Y1_row = []#read row in array struc
    flat_arr_images_Y1_row.append(tData[tData.columns[0]][n])
    flat_arr_images_Y1_matrix.append(flat_arr_images_Y1_row)
    n=n+1
flat_arr_images_X_matrix = numP.matrix(flat_arr_images_X_matrix,dtype='float64')
flat_arr_images_Y1_matrix = numP.matrix(flat_arr_images_Y1_matrix,dtype='float64')
Transform_flat_arr_images_X_matrix = flat_arr_images_X_matrix.T
dot_flat_arr_images_XandY_matrix= Transform_flat_arr_images_X_matrix * flat_arr_images_X_matrix
for i in range(dot_flat_arr_images_XandY_matrix[0].size):
    for j in range(dot_flat_arr_images_XandY_matrix[0].size):
        dot_flat_arr_images_XandY_matrix[i,j] =dot_flat_arr_images_XandY_matrix[i,j]+random.uniform(0, 1)#add w
inverse_dot_flat_arr_images_XandY_matrix = dot_flat_arr_images_XandY_matrix .I 
beta_Training_NN_Xt = inverse_dot_flat_arr_images_XandY_matrix * Transform_flat_arr_images_X_matrix
beta_Training1 = beta_Training_NN_Xt * flat_arr_images_Y1_matrix 
print("Beta1:",beta_Training1,beta_Training1.shape)
## real time camera to detect new person and crop its facial coordiante as test image.
video_capture = cv2.VideoCapture(0)
anterior = 0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass
    img_counter = 0
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(128, 188)
    )

## Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
##        print("x:",x,"\ty:",y,"\tx+w:",x+w,"\ty+h:",y+h)
        fp  = Image.open(img_name)
        cropped = fp.crop((x,y,x+w,y+h))
        cropped.save("cropped.png")
##        cropped.show()
##        print("{} written!".format(img_name))
        img_counter += 1
        numP.seterr(all='ignore')
        real_time_image = []  # all images in a folder for trainning X
        im = Image.open("cropped.png").resize((64, 64)).convert('RGBA')  # resize to smallest due memroy and matrix issue bestfit 64,64
        real_time_image.append(numP.array(im).ravel())  # here we can add bias,
        real_time_image = numP.matrix(real_time_image, dtype='float64')       
        print(real_time_image)
        predict_y = real_time_image* beta_Training1
        print("predict_value:",predict_y,"\n")
        if predict_y[0][0] > 0.3 and predict_y[0][0] < 3.3 :
            print("Aceess Granted, face matched.\n")
            break

    
    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


