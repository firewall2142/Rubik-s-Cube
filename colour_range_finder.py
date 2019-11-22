import cv2
import numpy as np
import matplotlib.pyplot as plt
from sq_detect import *
from random import randint
##
##try:
##    cap = cv2.VideoCapture(0)
##
##    
##    
##    while True:
##        ret, im = cap.read()
##        if not ret:
##            print 'bad camera !!!'
##            continue
####        im = cv2.medianBlur(im, 11)
##        cv2.imshow('camera input', im)
##        k = cv2.waitKey(1)
##        if k == ord('q'):
##            break
##        elif k == ord('s'):
##            filename = 'cap'+str(randint(0, 10000))+'.jpg'
##            cv2.imwrite(filename, im)
##            print 'saved ' + filename
##        elif k == ord('d'):
##            for attrib in [attr for attr in dir(cv2) \
##                if ('CAP_PROP' in attr and 0 < getattr(cv2, attr) < 20)]:
##                    print '%2d\t%20s\t%d'%(getattr(cv2, attrib),\
##                                           attrib, cap.get(getattr(cv2, attrib)))
##            print '----'*3 + '\n\n\n'
##        
##finally:
##    cap.release()
##    cv2.destroyAllWindows()

def mshow(im, titles = None):
    if str(type(im)) != str(type([])):
        plt.imshow(im)
        plt.show()
    else:
        m = int(pow(len(im), 0.5))
        n = int(math.ceil(len(im)/float(m)))
        for i in range(len(im)):
            plt.subplot(m, n, i+1)
            plt.imshow(im[i])
            if titles:
                plt.title(titles[i])
        plt.show()


im = cv2.imread('cap3765.jpg')
imrgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
imhsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
mshow([imrgb, imhsv], ['RGB', 'HSV'])
