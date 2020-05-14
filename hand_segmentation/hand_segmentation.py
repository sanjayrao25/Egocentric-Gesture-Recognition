import numpy as np
import argparse
import cv2
import os
from pyimagesearch import imutils



lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")
f=open("filelist.txt", 'a')
for dirpath, dirnames, files in os.walk('../ego_virtualmuseum/demo_room/subject_1'):
    print(f'Found directory: {dirpath}')
    trajectory=[[]]
    histOfGrad=[]
    histOfFlow=[]
    
    for file_name in files:
        #print(file_name)
        cap = cv2.VideoCapture("/media/akshay/New Volume/iiit_subjects/sem1/APS/Aps Projects/ego_virtualmuseum/demo_room/subject_1/"+file_name)
        #ret, frame1 = cap.read()
       
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4)) 
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out_filename="subject_1_out_"+file_name
        f.write("subject_1/"+out_filename+'\n')
        out = cv2.VideoWriter("/media/akshay/New Volume/iiit_subjects/sem1/APS/Aps Projects/ego_virtualmuseum/outputs/subject_1/"+out_filename,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
        #cv.imshow('frame1',frame1)
        #prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        #print(prvs.shape)
        #k = cv.waitKey(0)
        #hsv = np.zeros_like(frame1)
        #hsv[...,1] = 255
        #print(cap.isOpened())
        i=0;
        flag=True
        frame3count = 0;
        while True:
			# grab the current frame
            (grabbed, frame) = cap.read()
			# if we are viewing a video and we did not grab a
			# frame, then we have reached the end of the video
            if not grabbed:
                break
			
			# resize the frame, convert it to the HSV color space,
			# and determine the HSV pixel intensities that fall into
			# the speicifed upper and lower boundaries
            #frame = imutils.resize(frame, width = 400)
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            skinMask = cv2.inRange(converted, lower, upper)
			
			# apply a series of erosions and dilations to the mask
			# using an elliptical kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            skinMask = cv2.erode(skinMask, kernel, iterations = 2)
            skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
		
			# blur the mask to help remove noise, then apply the
			# mask to the frame
            skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
            skin = cv2.bitwise_and(frame, frame, mask = skinMask)
			
			# show the skin in the image along with the mask
            #cv2.imshow("images", np.hstack([frame, skin]))
            gray_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
            #cv2.imshow("gray", gray_image)
            #print(frame.shape)
            #print(gray_image.shape)
            out.write(skin)
			# if the 'q' key is pressed, stop the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
			
		# cleanup the camera and close any open windows
        cap.release()
        cv2.destroyAllWindows()
