from __future__ import print_function
import cv2 as cv
import argparse
import glob
import time


## [create]
#create Background Subtractor objects
if False:
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [create]

## [capture]
# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
# capture = cv.VideoCapture('vtest.avi')

org_folder = 'set2/';

images_org = glob.glob(org_folder+'*.jpg')

for fname in images_org:
    frame = cv.imread(fname)
    fgMask = backSub.apply(frame) 


for fname in images_org:
    frame = cv.imread(fname)
    fgMask = backSub.apply(frame)

    frame = cv.bitwise_and(frame,frame,mask = fgMask)

    ## [show]
    #show the current frame and the fg masks
    # cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    ## [show]
    # cv.waitKey(0)  

    fnameSplit = fname.split('\\')
    outFolder = 'masked_'+org_folder
    outName = fnameSplit[1]

    cv.imwrite(outFolder+outName,frame)

