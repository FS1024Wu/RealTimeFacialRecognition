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
def main():
    numP.seterr(all='ignore')
    flat_arr_images_X_matrix=[]#all images in a folder for trainning X
    flat_arr_images_Y1_matrix=[]#all images in a folder for trainning predict Y
    flat_arr_images_Y2_matrix=[]#all images in a folder for trainning predict Y
    flat_arr_images_XY_matrix = []
    n=0#iteration of scans
    tData = pd.read_excel(r"train.xlsx", sheet_name='Sheet1')
    print("Column headings:", tData.columns)
    for filename in glob.glob(r"train\*.PNG"):
        im=Image.open(filename).resize((64,64)).convert('RGBA')#resize to smallest due memroy and matrix issue bestfit 64，64
        flat_arr_images_X_matrix.append(numP.array(im).ravel())#  here we can add bias, 
        flat_arr_images_Y1_row = []#read row in array struc
        flat_arr_images_Y2_row = []#read row in array struc
        flat_arr_images_Y1_row.append(tData[tData.columns[0]][n])
        flat_arr_images_Y1_matrix.append(flat_arr_images_Y1_row)
        n=n+1
    flat_arr_images_X_matrix = numP.matrix(flat_arr_images_X_matrix,dtype='float64')
    x_rowSize = flat_arr_images_X_matrix[0].size
    flat_arr_images_Y1_matrix = numP.matrix(flat_arr_images_Y1_matrix,dtype='float64')
    Transform_flat_arr_images_X_matrix = flat_arr_images_X_matrix.T
    dot_flat_arr_images_XandY_matrix= Transform_flat_arr_images_X_matrix * flat_arr_images_X_matrix
    for i in range(dot_flat_arr_images_XandY_matrix[0].size):
        for j in range(dot_flat_arr_images_XandY_matrix[0].size):
              dot_flat_arr_images_XandY_matrix[i,j] =dot_flat_arr_images_XandY_matrix[i,j]+random.uniform(0, 1)#add w
    inverse_dot_flat_arr_images_XandY_matrix = dot_flat_arr_images_XandY_matrix .I 
    beta_Training_NN_Xt = inverse_dot_flat_arr_images_XandY_matrix * Transform_flat_arr_images_X_matrix
    beta_Training1 = beta_Training_NN_Xt * flat_arr_images_Y1_matrix 
##    beta_Training2 = beta_Training_NN_Xt * flat_arr_images_Y2_matrix
    print("Beta1:",beta_Training1,beta_Training1.shape)#,"\nBeta2:")#,beta_Training2,beta_Training2.shape,'\n')
# getting trainning model then to validate 
    test_image_X_matrix=[]
    test_image_Y1_matrix=[]
    test_image_Y2_matrix=[]
    n=0
    test_Data = pd.read_excel(r"C:\Users\sheng\Desktop\BDPFinal\train.xlsx", sheet_name='test')
    for filename in glob.glob(r"C:\Users\sheng\Desktop\BDPFinal\test\t2.*"):
        im=Image.open(filename).resize((64,64)).convert('RGBA') #resize to smallest due memroy and matrix issue
        test_image_X_matrix.append(numP.array(im).ravel())#  here we can add bias,
        test_image_Y1_row = []#read row in array struc
        test_image_Y1_row.append(test_Data[test_Data.columns[0]][n])
        test_image_Y1_matrix.append(test_image_Y1_row)
        n=n+1
    test_image_X_matrix = numP.matrix(test_image_X_matrix,dtype='float64')
    test_image_Y1_matrix = numP.matrix(test_image_Y1_matrix,dtype='float64')
    print("train_image_X for_Sheng:",test_image_X_matrix,test_image_X_matrix.shape,'\n')
    predict_image_Y1_matrix =  test_image_X_matrix * beta_Training1
    print("predict_model_Y for_Sheng:",predict_image_Y1_matrix )#, ",\tpredict_value_Y2 tobramycin resistance:",predict_image_Y2_matrix ,'\n')
    print("actual_image_value_Sheng:",test_image_Y1_matrix[0,0])#,",\tactual_value_Y2 tobramycin resistance:",test_image_Y2_matrix[0,0],'\n')    
    print("accuracy :", ((predict_image_Y1_matrix-test_image_Y1_matrix[0,0])/test_image_Y1_matrix[0,0])*100,"%,")#,
main()
