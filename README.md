# RealTimeFacialRecognition
* follow the step, you would be able to execute and see the result.
* prerequiment:  python IDLE 64 BITS. dependency python: glob, Pillow, xlrd, numpy, pandas so on make sure you have all of them you can find it in the code py file.
* question? contact info: fangshion@gmail.com Sheng Wu
* linear regression and logistic regreesion used, there are many algos you can use RBM, NN, DNN, I have my RBM and DNN coming soon.
*stay tuned.
* part of codes faceCascade and predefined facial model are open source 

1. unzip it to your local machine

2. open webcam_cv3.py change to your dictory 
such as: This is training dirctory
 tData = pd.read_excel(r"C:\Users\sheng\Desktop\train.xlsx", sheet_name='Sheet1')
    print("Column headings:", tData.columns)
    for filename in glob.glob(r"C:\Users\sheng\*.JPG"):

edit it to your path: 
 tData = pd.read_excel(r"YOURPATH\train.xlsx", sheet_name='Sheet1')
    print("Column headings:", tData.columns)
    for filename in glob.glob(r"YOURPATH\*.JPG"):
    
such as: This is testing dirctory
 test_Data = pd.read_excel(r"YOURPATH\train.xlsx", sheet_name='test')
    for filename in glob.glob(r"YOURPATH\test\t1.*"):
Step 2 applys to both Lin_Reg and Log_Reg python file

3. then run your code. minimum machine requires 2 Ghz processor with 12 G RAM.
	however, you can at line 26 and 70 edit it as: 
im=Image.open(filename).resize((56,56)).convert('RGBA')#resize to smallest due memroy and matrix issue bestfit 64,64
to 
im=Image.open(filename).resize((28,28)).convert('RGB')#resize to smallest due memroy and matrix issue bestfit 64,64
even 14,14 if you like. 

4. if somehow you get singular matrix error, add bias rand.random(0,1) just to make element in matrix nonzeor.

5. it is a simple model;thus, its accuracy not so high. There are way to imporve, logistic regression, fed more trainning images
and label corresponding 1 or 0 to the image is you(1) or not(0), more the better, this is a long traning time. 
Restricted Boltzmann machine and deep belief network, these two should be good fit for image processing. I am stilling learning those two.
model.
