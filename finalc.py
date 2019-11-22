import time
import serial
import kociemba
import cv2
import numpy as np
from sq_detect import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from operator import itemgetter

COM_PORT = 'COM3'


##color_lim = [(80,200,120),(90,255,255)] #mukesh bhaiya's cube green color
##color_lim = [(98,245,100),(105,255,150)]

#for mukesh's bhaiya's cube
##color_lims = {'G':(( 70,140, 65),( 90,255,255)), 'R':((165,210, 80),(  8,255,180)),
##              'W':((  0,  0, 90),(255, 80,255)), 'Y':(( 6,100,130),( 20,200,220)),
##              'B':((100,180, 70),(110,255,150)), }

##color_lims = {'G':(( 70,130,120),( 90,255,240)), 'R':((170,110,110),(10,255,190)),
##              'W':((  0,  0, 160),(255, 50,255)), 'Y':(( 30, 80,150),( 50,130,255)),
##              'B':(( 95,120,120),(105,255,220)), 'O':((175, 180, 190),(10,255,255))}

##color_lims = {'G':(( 70,120,120),( 90,255,240)), 'R':((160,80,90),(20,255,190)),
##              'W':((  70,  0, 135),(120, 100,255)), 'Y':(( 30, 60,140),( 60,190,255)),
##              'B':(( 95,110,110),(105,255,230)), 'O':((170, 110, 180),(10,255,255))}


##color_lims = {'G':(( 70,120,100),( 90,255,240)), 'R':((160,110,90),(20,255,160)),
##              'W':((  70,  0, 135),(120, 100,255)), 'Y':(( 30, 60,140),( 60,190,255)),
##              'B':(( 95,110,110),(105,255,230)), 'O':((170, 110, 190),(10,255,255))}

color_lims = {'G':(( 70,120,70),( 95,255,240)), 'R':((160,110,90),(15,255,160)),
              'W':((  0,  0, 135),(255, 100,255)), 'Y':(( 20, 60,90),( 60,190,255)),
              'B':(( 95,110,60),(120,255,230)), 'O':((170, 110, 160),(10,255,255))}


DS_SQUARE_SIDE_RATIO = 1.2
DS_MORPH_KERNEL_SIZE = 5
DS_MORPH_ITERATIONS = 2
DS_MIN_SQUARE_LENGTH_RATIO = 0.08
DS_MIN_AREA_RATIO = 0.7
DS_MIN_SQUARE_SIZE = 0.05 #times the width of image
DS_MAX_SQUARE_SIZE = 0.25

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
        

        
def cvshow(im, win=''):
    cv2.imshow(win, im)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()

#pts = [(x,y), ...]
#returns mat[3][3] = [[index1, index2, index3], ...]
#if unable to do it returns None
def index_to_cube(pts):

    if len(pts) != 9:
        return None
    
    pts = [list(pts[i])+[i] for i in range(len(pts))]
    pts.sort(key=itemgetter(1))
    mat = [[pts[3*i+j] for j in range(3)] for i in range(3)]
    for i in range(3):
        mat[i].sort(key=itemgetter(0))

    for i in range(3):
        for j in range(3):
            mat[i][j] = mat[i][j][2]

    return mat


def get_color_mask(im, color_name, smoothed=True, isBGR=False):

    DEBUG_SHOW_MASK = False

    if isBGR:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    color_lim = color_lims[color_name]
    
    if color_name in 'RO':
        #((165,210, 80),(  8,255,180))
        big_hue = max((color_lim[0][0], color_lim[1][0]))
        small_hue = min((color_lim[0][0], color_lim[1][0]))
        
        lower_red = [(0, color_lim[0][1], color_lim[0][2]), (small_hue, color_lim[1][1], color_lim[1][2])]
        upper_red = [(big_hue, color_lim[0][1], color_lim[0][2]), (180, color_lim[1][1], color_lim[1][2])]

        mask = cv2.inRange(im, upper_red[0], upper_red[1])|cv2.inRange(im, lower_red[0], lower_red[1])
        
##        mask = ((im[:,:,0]>=big_hue|(im[:,:,0]<=small_hue)&\
##               (im[:,:,1]>=color_lim[0][1])&(im[:,:,1]<=color_lim[1][1])&\
##               (im[:,:,2]>=color_lim[0][2])&(im[:,:,2]<=color_lim[1][2])))
##        mask = np.array(mask, dtype=np.uint8)*255

    else:
        mask = cv2.inRange(im, color_lim[0], color_lim[1])
    if smoothed:
        kernel = np.ones(tuple([DS_MORPH_KERNEL_SIZE]*2))
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1) 

    if DEBUG_SHOW_MASK:
        mshow([mask], [color_name + ' mask in function'])

    return mask
            
#accepts BGR image
def detect_square(im, color_name):

    DEBUG_SHOW_MASK = False
    DEBUG_SHOW_INSIDE_FUNC = False

    def remove_bad_contours(conts):
        new_conts = []
        
        for cont in conts:
            bound_rect = cv2.minAreaRect(cont)
            length, breadth = float(bound_rect[1][0]), float(bound_rect[1][1])
            try:
##                print length/breadth, cv2.contourArea(cont)/(length*breadth)
                if max((length/breadth, breadth/length)) > DS_SQUARE_SIDE_RATIO:
                    continue
                if cv2.contourArea(cont)/(length*breadth) < DS_MIN_AREA_RATIO:
                    continue
                if not DS_MAX_SQUARE_SIZE*im.shape[0] > max((length, breadth)) > DS_MIN_SQUARE_SIZE*im.shape[0]:
                    continue
##                print length/breadth, cv2.contourArea(cont)/(length*breadth)
                new_conts.append(cont)
            except ZeroDivisionError:
                continue

        return new_conts


    im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    global debug_mask

    mask = get_color_mask(im, color_name)

    debug_mask=[np.array(mask)]

    if DEBUG_SHOW_MASK:
        mshow(debug_mask, ['COLOUR ' + color_name]*len(debug_mask))
    
    conts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    
    conts = remove_bad_contours(conts)

    if DEBUG_SHOW_INSIDE_FUNC:
        im2 = np.array(im)
        for cont in conts:
            cv2.circle(im2, tuple(np.array(cv2.minAreaRect(cont)[0],dtype=int)),
                       int(pow(cv2.contourArea(cont)/3.14159, 0.5)), (0,255,255),thickness=2)
        mshow(cv2.cvtColor(im2, cv2.COLOR_HSV2RGB))

    return [cv2.minAreaRect(cont) for cont in conts]



#accepts BGR
def get_cube_state(im):

    colors_detected = []
    cube_state = [[None]*3 for _ in range(3)]
    
    for color, color_lim in zip(color_lims.keys(), color_lims.values()):
        rects = detect_square(im, color)

        for rect in rects:
            colors_detected.append((color, rect))
        
    #from pprint import pprint
    #pprint(colors_detected)

    index_mat = index_to_cube([prop[1][0] for prop in colors_detected])

    
    if index_mat != None:
        for i in range(3):
            for j in range(3):
                cube_state[i][j] = colors_detected[index_mat[i][j]][0]
                cube_state[i][j] = cube_state[i][j].lower()

        return cube_state
    else:
        return None


#accepts BGR image and BGR or HSV colour
def draw_rects(im, rects, colour, iscolHSV=True):

    im2 = np.array(im)

    if iscolHSV:
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
    
    boxpts = np.array([cv2.boxPoints(rect) for rect in rects], dtype=np.int32)
    cv2.polylines(im2, boxpts, True, colour, thickness=5)
    
    return cv2.cvtColor(im2, cv2.COLOR_HSV2BGR)


    
def sol(input1):
	input2=[[[]],[[]],[[]],[[]],[[]],[[]]]

	for i in range(6):
		if(input1[i][1][1]=='w'):
			input2[0]=input1[i]
			break

	for i in range(6):
		if(input1[i][1][1]=='r'):
			input2[1]=input1[i]
			break

	for i in range(6):
		if(input1[i][1][1]=='g'):
			input2[2]=input1[i]
			break

	for i in range(6):
		if(input1[i][1][1]=='y'):
			input2[3]=input1[i]
			break


	for i in range(6):
		if(input1[i][1][1]=='o'):
			input2[4]=input1[i]
			break


	for i in range(6):
		if(input1[i][1][1]=='b'):
			input2[5]=input1[i]
			break

	for i in range(6):
		for j in range(3):
			for k in range(3):
				if (input2[i][j][k]=='w'):
					input2[i][j][k]='U'
				elif (input2[i][j][k]=='y'):
					input2[i][j][k]='D'
				elif (input2[i][j][k]=='r'):
					input2[i][j][k]='R'
				elif (input2[i][j][k]=='o'):
					input2[i][j][k]='L'
				elif (input2[i][j][k]=='g'):
					input2[i][j][k]='F'
				elif(input2[i][j][k]=='b'):
					input2[i][j][k]='B'

	b=''
	for i in range(6):
		for j in range(3):
			for k in range(3):
				b+=input2[i][j][k]
	
       
	a = kociemba.solve(b)
	print(a)
	
	a = a.replace("U'","R L F2 B2 R' L' D' R L F2 B2 R' L'")
	a = a.replace("U2","R L F2 B2 R' L' D2 R L F2 B2 R' L'")
	a = a.replace("U","R L F2 B2 R' L' D R L F2 B2 R' L'")
	a=a.replace("D2","D R R' D")#REMOVE!!!!!!!!!!!!1
	return a

def cleansolution(solution):
    cleanedsolution=""
    prev=solution[0]
    for current in solution[1:]:
        if current=="'":
            cleanedsolution+=prev.lower()
        elif current=="2":
            cleanedsolution+=prev
            cleanedsolution+=prev
        else:
            cleanedsolution+=prev
        prev=current
    cleanedsolution = cleanedsolution.replace("'", "")
    cleanedsolution = cleanedsolution.replace("2", "")
    cleanedsolution = cleanedsolution.replace(" ", "")
    
    return cleanedsolution

def sendarduino(solution):
    global ser
    ser = serial.Serial(COM_PORT, 9600,timeout=None) # Establish the connection on a specific port
    time.sleep(1)
    count=0
    for char in solution:
        ser.write(char.encode())
        time.sleep(1)
        #count+=1
        #if count==60:
        #    time.sleep(60)#CHANGE SLEEP TIME ACCORDING TO DELAY BETWEEN MOTORS
        #    count=0
    ser.close()


try:

    SCALE=1.3
    
    cap = cv2.VideoCapture(0)
    k = ''

    cube_state = []
    while True:
        im = cap.read()[1]

        disp_im = np.array(im)
        for color in color_lims:
            sq = detect_square(im, color)
            disp_im = draw_rects(disp_im, sq, color_lims[color][1])

        disp_im = cv2.resize(disp_im, (1280,960))
        disp_im = cv2.flip(disp_im, 1)
        
        cv2.imshow('detected_cube', disp_im)
        
        k = cv2.waitKey(1)
        if k == ord('q') or k == ord('d'):
            break
        if k == ord('s'):
            state= get_cube_state(im)
            if state != None:
                flatten = ''.join(sum(sum(cube_state, []), []))
                for col in color_lims:
                    col = col.lower()
                    if flatten.count(col) > 9:
                        cube_state = []
                        print flatten.count(col), 'occurences of', col
                cube_state.append(state)
                print 'added', state
            else:
                print 'invalid faces'
            if len(cube_state) == 6:
                print 'cube state = \n',cube_state
                solution = cleansolution(sol(list(cube_state)))
                print 'steps = ', len(solution)
                cap.release()
                print solution
                sendarduino(solution)
                break
        elif k == ord('r'):
                print 'reset cube'
                cube_state = []
        
finally:
    cap.release()
    cv2.destroyAllWindows()

    if k == ord('d'):
        images = []
        titles = []
        for color in color_lims:
            images.append(get_color_mask(im, color, smoothed=False, isBGR=True))
            titles.append(color)
        images.append(cv2.cvtColor(disp_im, cv2.COLOR_BGR2RGB))
        titles.append('RGB')
        images.append(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        titles.append('HSV')
        cv2.imwrite('testim.jpg', im)
        mshow(images, titles)


#DrlBFrllbfdLbbrff
