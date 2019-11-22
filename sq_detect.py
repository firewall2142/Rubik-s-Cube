import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from operator import itemgetter



##color_lim = [(80,200,120),(90,255,255)] #mukesh bhaiya's cube green color
##color_lim = [(98,245,100),(105,255,150)]

#for mukesh's bhaiya's cube
##color_lims = {'G':(( 70,140, 65),( 90,255,255)), 'R':((165,210, 80),(  8,255,180)),
##              'W':((  0,  0, 90),(255, 80,255)), 'Y':(( 6,100,130),( 20,200,220)),
##              'B':((100,180, 70),(110,255,150)), }

color_lims = {'G':(( 50,100,100),( 90,255,240)), 'R':((170,110,110),(10,255,155)),
              'W':((  0,  0, 120),(255, 50,255)), 'Y':(( 11, 80,150),( 40,255,255)),
              'B':(( 95,60,70),(120,255,220)), 'O':((175, 180, 155),(10,255,255))}

##color_lims = {'G':(( 70,120,120),( 90,255,240)), 'R':((170,100,100),(10,255,190)),
##              'W':((  0,  0, 150),(255, 50,255)), 'Y':(( 30, 70,140),( 50,190,255)),
##              'B':(( 95,110,110),(105,255,230)), 'O':((175, 110, 180),(10,255,255))}



DS_SQUARE_SIDE_RATIO = 1.5
DS_MORPH_KERNEL_SIZE = 5
DS_MORPH_ITERATIONS = 2
DS_MIN_SQUARE_LENGTH_RATIO = 0.08
DS_MIN_AREA_RATIO = 0.7
DS_MIN_SQUARE_SIZE = 0.10 #times the width of image
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

def getColor(char):
    m = {'r': (255,0,0), 'o':(255,128,0), 'b':(0,0,255),
         'g':(0,255,0), 'w':(255,255,255), 'y':(255,255,0)}
    return tuple(list(m[char.lower()])[::-1])

def draw_cube(im, cube_state):
    
    for i in range(3):
        for j in range(3):
            if cube_state != None:
                im = cv2.rectangle(im, (10+40*j, 10+40*i),
                                   (40+40*j, 40+40*i), getColor(cube_state[i][j]), 3)
                im = cv2.putText(im, cube_state[i][j], (17+40*j, 32+40*i),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                 getColor(cube_state[i][j]), 1)
            else:
                im = cv2.rectangle(im, (10+40*j, 10+40*i),
                                   (40+40*j, 40+40*i), (128,128,128), 3)
                im = cv2.putText(im, '?', (17+40*j, 32+40*i),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                 (128,128,128), 1)

    return im



if __name__ == '__main__':
##    im = cv2.imread('cap3765.jpg')
##
##    COLOUR = 'B'
##    rects= detect_square(im, COLOUR)
##    print 'rects : ', rects
##    boxpts = np.array([cv2.boxPoints(rect) for rect in rects], dtype=np.int32)
##    cv2.polylines(im, boxpts, True, (0, 255, 255))
##    cvshow(im, 'result')
##    if 'y' in raw_input('\nshow HSV'):
##        mshow([cv2.cvtColor(im, cv2.COLOR_BGR2HSV), cv2.cvtColor(im, cv2.COLOR_BGR2RGB)])

    try:
        cap = cv2.VideoCapture(0)
        im = cap.read()[1]
        fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
        from random import randint
        filename = str(randint(1,10000)) + '.avi'
        out = cv2.VideoWriter(filename, fourcc, 10, (640,480))
        print 'saving as ', filename
        k = ''
        
        while True:
            im = cap.read()[1]
            disp_im = np.array(im)
            for color in color_lims:
                sq = detect_square(im, color)
                disp_im = draw_rects(disp_im, sq, color_lims[color][1])

            cube_state = get_cube_state(im)
            disp_im = draw_cube(disp_im, cube_state)
            out.write(disp_im)
            cv2.imshow('final', disp_im)
            k = cv2.waitKey(1)
            if k == ord('q') or k == ord('d'):
                break
            if k == ord('s'):
                print get_cube_state(im)
    except Exception as e:
        print e
    finally:
        cap.release()
        out.release()
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
        





