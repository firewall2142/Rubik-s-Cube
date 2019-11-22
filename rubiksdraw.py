import cv2
import numpy as np

## +--+--+ (10, 10), (40, 10), (
## |  |  |
## +--+--+
## |  |  |
## +--+--+

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


im = np.zeros((640,480,3), dtype=np.uint8)
cube_state = [['R','G', 'B'],['W','Y','G'],['O','R','Y']]
cube_state = None

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
cv2.imshow('',im)
cv2.waitKey(-1)
cv2.destroyAllWindows()
