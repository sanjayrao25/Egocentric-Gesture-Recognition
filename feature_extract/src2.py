import cv2 as cv
import os   # for directory reading
import numpy as np
from numpy.linalg import norm
from histogramOfOpticalFlow import hof


entries = os.scandir('../ego_virtualmuseum/outputs')
stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30, 0.001)
binSize=180/8
patchSize=32

################## HOG init ####################


winSize = (32,32)#(64,64)
blockSize = (16,16)#(16,16)
blockStride = (16,16)#(8,8)
cellSize = (16,16)#(8,8)
nbins = 8
derivAperture = 1
winSigma = 4.
histogramNormType = 0 #HOGDescriptor::L2Hys
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels =  32 #HOGDescriptor::DEFAULT_NLEVELS
hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma, histogramNormType,L2HysThreshold,gammaCorrection,nlevels)





###############################################
for dirpath, dirnames, files in os.walk('../ego_virtualmuseum/outputs/subject_1_1'):
    print(f'Found directory: {dirpath}')
    trajectory=[[]]
    histOfGrad=[]
    histOfFlow=[]
    for file_name in files:
        print(file_name)
        cap = cv.VideoCapture("/media/akshay/New Volume/iiit_subjects/sem1/APS/Aps Projects/ego_virtualmuseum/outputs/subject_1_1/"+file_name)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4)) 
        out_filename="subject_1_out_"+file_name
        out = cv.VideoWriter("/media/akshay/New Volume/iiit_subjects/sem1/APS/Aps Projects/ego_virtualmuseum/outputs/"+out_filename,cv.VideoWriter_fourcc('H','2','6','4'), 10, (frame_width,frame_height))
        ret, frame1 = cap.read()
        #cv.imshow('frame1',frame1)
        prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
        #print(prvs.shape)
        #k = cv.waitKey(0)
        #hsv = np.zeros_like(frame1)
        #hsv[...,1] = 255
        #print(cap.isOpened())
        i=0;
        flag=True
        frame3count = 0;
        framecount=0
        while(cap.isOpened()):
            ret, frame2 = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            nextgray = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
            if flag :
                corners = cv.goodFeaturesToTrack(prvs,0,0.001,minDistance=5,blockSize=5,useHarrisDetector=False)
                cv.cornerSubPix(prvs, corners, (2,2), (-1,-1), stop_criteria)
                flag=False
            
            flow = cv.calcOpticalFlowFarneback(prvs,nextgray, None, 0.707, 8, 5, 3, 5, 1.2, 0) # Pyramid_scale = 1/sqrt(2), No. of spatial scales=8(levels), winSize=5 (15)
            medianflow = cv.medianBlur(flow, ksize=3)
            #print(flow.shape)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            
            #print(corners.shape)
            currentTraj=[]
            sumx=0
            sumy=0
            num_rows, num_cols = nextgray.shape
            for i in corners:
                p,q = int(i[0][1]),int(i[0][0])
                if p!=-1 or q!=-1 :
                    currentTraj.append((medianflow[p][q][0], medianflow[p][q][1]))
                    i[0][1],i[0][0] =  (p+medianflow[p][q][0]), (q+medianflow[p][q][1])
                if (i[0][1]>=num_rows-32) or (i[0][1]<16) or (i[0][0]>=num_cols-32) or (i[0][0]<16):        # 32x32 is the patch size for calculating HoG, HoF, MBH
                    i[0][0],i[0][1] = -1,-1


                magnitude=mag[int(i[0][1])][int(i[0][0])]
                angle=ang[int(i[0][1])][int(i[0][0])]
                
                x =  round(i[0][0] + magnitude * np.cos(angle));
                y =  round(i[0][1] + magnitude * np.sin(angle));
                xdisp=x-int(i[0][0])
                ydisp=y-int(i[0][1])
                sumx=sumx+abs(xdisp)
                sumy=sumy+abs(ydisp)
                cv.line(frame2, (int(i[0][0]),int(i[0][1])),(int(x),int(y)), (0,255,0),2)
                
            trajectory.append(currentTraj)

            #for i,j in zip(corners, )
            #nextPoints= goodfeatures to track + median
            #cv.imshow('frame2',bgr)
            #cv.moveWindow('frame2', 1200,1200)
            #cv.waitKey(5)
            #cv.imshow('opticalfb',frame2)
            #cv.waitKey(5)
            prvs = nextgray

        
            #     cv.imwrite('opticalhsv.png',bgr)
            
################################## HOG ###################################
            cuboidHist = [[[]]]
            
            for i in corners:
                p,q = int(i[0][1]),int(i[0][0])
                patch_center = (p,q)
                patch_size = 32
                imgHist = [[]]
                if p!=-1 and q!=-1:
                    patch_x = int(patch_center[0] - patch_size / 2.)
                    patch_y = int(patch_center[1] - patch_size / 2.)
                    hog_patch_image = nextgray[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
                    hog_hist = hog.compute(hog_patch_image)
                    hog_hist = hog_hist.flatten()
                    imgHist.append(hog_hist)
                    #hof_patch = flow[patch_x:patch_x+patch_size, patch_y:patch_y+patch_size]
                    #hof_hist = hog.compute(hof_patch)
                    #hof_hist = hof_hist.ravel()
                    #print(hof_hist)
                    
                else :
                    temp = []
                    imgHist.append(temp)

            cuboidHist.append(imgHist)
            frame3count+=1
            if frame3count == 3:
                # Flatten cuboidHist to form 96x1 descriptor, store it and clear it for next 3 frames
                cuboidHist = [[[]]]
#########################################################################
                
########################## HoF ##########################################
            histOfFlow, hof_image=hof(flow,9,(16,16),(2,2), True)
            cv.imshow('hof_image',hof_image)
            framecount+=1
            cv.imwrite('hof_image_'+str(framecount)+'.jpg', hof_image)
            arr = np.uint8(hof_image)
            color_hof = cv.cvtColor(arr, cv.COLOR_GRAY2RGB)
            #print(hof_image)
            out.write(color_hof)
            cv.waitKey(5)
            #print(histOfFlow)
            
#########################################################################
            xflow = flow[...,0]
            yflow = flow[...,1]
            print(xflow.shape)
            print("done")
            flowXdX = cv.Sobel(xflow, cv.CV_32F, 1, 0, 1);
            flowXdY = cv.Sobel(xflow, cv.CV_32F, 0, 1, 1);
            flowYdX = cv.Sobel(yflow, cv.CV_32F, 1, 0, 1);
            flowYdY = cv.Sobel(yflow, cv.CV_32F, 0, 1, 1);
        cap.release()
        cv.destroyAllWindows()
