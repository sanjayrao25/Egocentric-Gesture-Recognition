import cv2 as cv
import os   # for directory reading
import numpy as np
from numpy.linalg import norm

entries = os.scandir('../ego_virtualmuseum/demo_room')
stop_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30, 0.001)
binSize=180/8
patchSize=32
for dirpath, dirnames, files in os.walk('../ego_virtualmuseum/demo_room/subject_1'):
    print(f'Found directory: {dirpath}')
    trajectory=[[]]
    histOfGrad=[]
    histOfFlow=[]
    for file_name in files:
        #print(file_name)
        cap = cv.VideoCapture("/media/akshay/New Volume/iiit_subjects/sem1/APS/Aps Projects/ego_virtualmuseum/demo_room/subject_1/"+file_name)
        ret, frame1 = cap.read()
        #cv.imshow('frame1',frame1)
        prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
        #k = cv.waitKey(0)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        print(cap.isOpened())
        i=0;
        while(cap.isOpened()):
            ret, frame2 = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)


            corners = cv.goodFeaturesToTrack(next,0,0.001,minDistance=10,blockSize=5,useHarrisDetector=False)
            cv.cornerSubPix(next, corners, (2,2), (-1,-1), stop_criteria)
            #print(corners.shape)
            
            
            flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.707, 8, 5, 3, 5, 1.2, 0) # Pyramid_scale = 1/sqrt(2), No. of spatial scales=8(levels), winSize=5 (15)
            medianflow = cv.medianBlur(flow, ksize=3)
            #print(flow.shape)
            mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
            #print(mag.shape,   ang.shape)
            hsv[...,0] = (ang*180/(np.pi/2))
            hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
            currentTraj=[]
            sumx=0
            sumy=0
            for i in corners:
                #print(j,k)
                magnitude=mag[int(i[0][1])][int(i[0][0])]
                angle=ang[int(i[0][1])][int(i[0][0])]
                
                x =  round(i[0][0] + magnitude * np.cos(angle));
                y =  round(i[0][1] + magnitude * np.sin(angle));
                xdisp=x-int(i[0][0])
                ydisp=y-int(i[0][1])
                sumx=sumx+abs(xdisp)
                sumy=sumy+abs(ydisp)
                currentTraj.append((xdisp,ydisp))
                cv.line(frame2, (int(i[0][0]),int(i[0][1])),(int(x),int(y)), (0,255,0))
                #cv.circle( frame2, (int(i[0][0]),int(i[0][1])), 3, (0,255,0), -1)
            print(currentTraj)
            currentTraj=np.array(currentTraj)
            for i in currentTraj:
                i[0]/=sumx
                i[1]/=sumy
            print(currentTraj)
            trajectory.append(currentTraj)

            for i in corners:
                if ((i[0][0] < patchSize/2) or (i[0][0] > (frame2.cols-patchSize/2-1)) or (i[0][1] < patchSize/2) or (i[0][1] > (frame2.rows-patchSize/2-1))):
                    continue

            currentHog=np.zeros(8)
            currentHof=np.zeros(8)
            for i in corners:
                magnitude=mag[int(i[0][1])][int(i[0][0])]
                angle=ang[int(i[0][1])][int(i[0][0])]
                angDeg=(ang*180/(np.pi/2))
                binVal=angDeg/8
                modVal=binVal%(22.5)
                index=int(angDeg/binSize)
                if modVal < (binSize/2) :
                    currentTraj[index+1]+=magnitude*(binSize-modVal)/binSize
                elif modVal > (binSize/2) :
                    currentTraj[index]+=magnitude*modVal/binSize
                else:
                    currentTraj[index]+=magnitude
                norm(currentTraj)

            
            histOfGrad.append(currentTraj)
            histOfFlow.append()

            cv.imshow('frame2',bgr)
            cv.moveWindow('frame2', 1200,1200)
            cv.waitKey(5)
            cv.imshow('opticalfb',frame2)
            cv.waitKey(5)

           #     cv.imwrite('opticalhsv.png',bgr)
            prvs = next

        cap.release()
        cv.destroyAllWindows()

