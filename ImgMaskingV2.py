# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import random
import numpy as np
import glob
import piexif

def createMask(img1, img2):
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--first", required=False,
        help="first input image")
    ap.add_argument("-s", "--second", required=False,
        help="second")
    args = vars(ap.parse_args())

    # load the two input images
    # imageA = cv2.imread(args["first"])
    # imageB = cv2.imread(args["second"])
    tmp = cv2.imread(img1)
    orgWidth = tmp.shape[1]
    orgHeight = tmp.shape[0]
    scale_percent = 20 # percent of original size
    width = int(orgWidth * scale_percent / 100)
    height = int(orgHeight * scale_percent / 100)
    orgDim = (orgWidth, orgHeight)
    dim = (width, height)
    

    imageA = cv2.resize(cv2.imread(img1),dim)
    imageB = cv2.resize(cv2.imread(img2),dim)

    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("SSIM: {}".format(score))

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    biggestCntArea = 0
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour and then draw the
        # bounding box on both input images to represent where the two
        # images differ
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cArea = w*h

        if cArea > biggestCntArea:
            biggestCntArea = cArea
            biggestCnt = c
    
    # draw all the contours
    for c in cnts:
        r = random.randint(0,256)
        g = random.randint(0,256)
        b = random.randint(0,256)
        cv2.drawContours(imageA,[c],0,(r,g,b),1)
        cv2.drawContours(imageB,[c],0,(r,g,b),1)

    # create mask from the biggest contour
    mask = np.zeros((height,width),dtype=np.uint8)
    cv2.fillPoly(mask,pts =[biggestCnt], color=255)

    # show the output images
    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)

    maskOut = cv2.resize(mask,orgDim,interpolation=cv2.INTER_LINEAR)
    return maskOut

def maskImage(inFolder,outFolder,bgFile):
    imOrg = glob.glob(inFolder+'*.jpg')

    for fname in imOrg:
        fnameOut = fname.split('\\')[1]
        fname_out = outFolder+fnameOut
        img = cv2.imread(fname)

        orgWidth = img.shape[1]
        orgHeight = img.shape[0]
        scale_percent = 50 # percent of original size
        width = int(orgWidth * scale_percent / 100)
        height = int(orgHeight * scale_percent / 100)
        orgDim = (orgWidth, orgHeight)
        dim = (width, height)
        
        fgMask = createMask(fname,bgFile)
        
        # fgMaskCrop = cv2.resize(fgMask,dim)
        # imgCrop = cv2.resize(fgMask,dim)
        # cv2.imshow('original',cv2.resize(img,dim))
        # cv2.imshow('mask',cv2.resize(fgMask,dim))
        # cv2.waitKey(0)
        imMasked = cv2.bitwise_and(img,img,mask=fgMask)
        

        # cv2.imshow('original',cv2.resize(img,dim))
        # cv2.imshow('masked',cv2.resize(imMasked,dim))
        # cv2.imshow('mask',cv2.resize(fgMask,dim))
        # cv2.waitKey(0)
        # imMasked_resise = cv2.resize(imMasked,dim)

        cv2.imwrite(fname_out,imMasked)
        try:
            piexif.transplant(fname,fname_out)
        except:
            print("Could not transplant EXIF information for image %s" %fnameOut)

        print("Image %s masked " %fnameOut)
        

maskImage('input/','masked/','bg.jpg')
