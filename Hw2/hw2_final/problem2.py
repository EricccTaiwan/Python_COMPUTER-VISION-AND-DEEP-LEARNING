import cv2
import numpy as np
import os
import time
import glob

def cornerDetection() :
    cap = cv2.VideoCapture(r"C:\Users\Windows\Desktop\hw2\1.avi")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(5) == ord('0'):
            break
    cap.release()
    cv2.destroyAllWindows()

def intrinsicMatrix():
    CHECKERBOARD = (11,8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = [] 
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    images = glob.glob(r"C:\Users\Windows\Desktop\hw2\Q2_Image\*.bmp")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	    cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,8),(-1,-1),criteria) 
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print("Camera matrix : \n")
    print(mtx)

def distortionMatrix():
    CHECKERBOARD = (11,8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = [] 
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None
    images = glob.glob(r"C:\Users\Windows\Desktop\hw2\Q2_Image\*.bmp")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+
    	    cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,8),(-1,-1),criteria) 
            imgpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2,ret)
    cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    print("Distortion matrix : \n")
    print(dist)

def findextrinsic(num):
	img=[]
	objp = np.zeros((8*11,3), np.float32)
	objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
	objpoints=[]
	imgpoints=[]
	for i in range(0,15):
		filename=i+1
		filename=str(filename)
		filename=filename+".bmp"
		filename="C:\\Users\\Windows\\Desktop\\hw2\\Q2_Image\\"+filename
		temp=cv2.imread(filename)
		img.append(temp)
	imgedit=[]
	for i in range(0,15):
		ret,corners=cv2.findChessboardCorners(img[i], (11, 8),None)
		if ret==True:
			objpoints.append(objp)
			temp = cv2.drawChessboardCorners(img[i], (11, 8), corners, ret)
			imgedit.append(temp)
			imgpoints.append(corners)
	gray=cv2.cvtColor(img[0],cv2.COLOR_BGR2GRAY)
	number=int(num)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	rotation=(rvecs[number-1][0][0],rvecs[number-1][1][0],rvecs[number-1][2][0])
	rotation=cv2.Rodrigues(rotation)[0]
	extrinsic=np.c_[rotation,tvecs[number-1]]
	print("Extrinsic Matrix:\n",extrinsic)

def undistort():
    cap = cv2.VideoCapture(r"C:\Users\Windows\Desktop\distort_1.avi")
    cap2 =cv2.VideoCapture(r"C:\Users\Windows\Desktop\undistort_1.avi")
    while cap.isOpened():
        ret, frame = cap.read()
        ret,frame2=cap2.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('distort', frame)
        cv2.imshow('undsitort',frame2)
        if cv2.waitKey(5) == ord('0'):
            break
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
