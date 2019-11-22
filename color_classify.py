import cv2
import numpy as np

#white -> low sat, high val
#yellow -> 60
#blue -> 240
#green -> 120
#red -> 0
#orange -> 30


#accepts HSV
def classify(colors):

    N = 6*9
    
    def remove_indices(arr, ind):
        arr2 = [ele for ele in arr if ele[1] not in ind]
        return arr2

    colors = np.array(colors)
    
    colour_index = {'white': [], 'green': [], 'blue': [],
                'red':[], 'yellow':[], 'orange':[]}
    
    hue = zip(colors[:, 0].tolist(), range(N))
    sat = zip(colors[:, 1].tolist(), range(N))
    val = zip(colors[:, 2].tolist(), range(N))

    hue.sort()
    sat.sort()
    val.sort()

    #whites
    sat.reverse()

    for i in range(N/6):
        color = sat[i]
        colour_index['white'].append(color[1])

    hue = remove_indices(hue, colour_index['white'])
    sat = remove_indices(sat, colour_index['white'])
    val = remove_indices(val, colour_index['white'])

    #red
    dist_index = np.argsort(np.abs(colors[:,0]))

    for i in range(N/6):
        colour_index['red'].append(dist_index[i])

    hue = remove_indices(hue, colour_index['red'])
    sat = remove_indices(sat, colour_index['red'])
    val = remove_indices(val, colour_index['red'])

    del dist_index

    #orange
    hue.sort()

    for i in range(N/6):
        colour_index['orange'].append(hue[i][1])

    hue = remove_indices(hue, colour_index['orange'])
    sat = remove_indices(sat, colour_index['orange'])
    val = remove_indices(val, colour_index['orange'])

    #yellow
    hue.sort()

    for i in range(N/6):
        colour_index['yellow'].append(hue[i][1])

    hue = remove_indices(hue, colour_index['yellow'])
    sat = remove_indices(sat, colour_index['yellow'])
    val = remove_indices(val, colour_index['yellow'])

    #green
    hue.sort()

    for i in range(N/6):
        colour_index['green'].append(hue[i][1])

    hue = remove_indices(hue, colour_index['green'])
    sat = remove_indices(sat, colour_index['green'])
    val = remove_indices(val, colour_index['green'])

    #blue
    hue.sort()

    for i in range(N/6):
        colour_index['blue'].append(hue[i][1])

    hue = remove_indices(hue, colour_index['blue'])
    sat = remove_indices(sat, colour_index['blue'])
    val = remove_indices(val, colour_index['blue'])

    output_colors = [0]*N

    for col in colour_index:
        l = colour_index[col]
        for i in l:
            output_colors[i] = col[0].upper()
    
    return output_colors



'''
def webcam_input():

    ROI_RATIO = 0.6
    ROI_RECT_COLOUR = (0, 255, 0)

    try:    
        cap = cv2.VideoCapture(0)

        for _ in range(10):
            _, capimg = cap.read()
            
        WIDTH = capimg.shape[1]
        HEIGHT = capimg.shape[0]
        CENTER = np.array((HEIGHT, WIDTH), dtype=int)/2

        def read_colors_from_roi(roi):
            WIDTH = roi.shape[1]
            HEIGHT = roi.shape[0]

            xspace = np.linspace(0, WIDTH, 4, dtype=int)
            yspace = np.linspace(0, HEIGHT, 4, dtype=int)
            
            col_mat = np.zeros((3,3,3), dtype=int)

            for i in range(3):
                for j in range(3):
                    cur_roi = roi[yspace[j]:yspace[j+1], xspace[i]:xspace[i+1], :]

                    median_col = [0, 0, 0]
                    
                    for k in range(3):
                        median_col[k] = np.median(cur_roi[:,:,k].ravel())
                    
                    col_mat[i,j,:] = np.array(median_col, dtype=int)


            return col_mat

        SQUARE_DELTA = np.array([int(HEIGHT*ROI_RATIO/2)]*2, dtype=int) #HALFED square size

        Y0, X0, Y3, X3 = (CENTER - SQUARE_DELTA).tolist() + (CENTER + SQUARE_DELTA).tolist()

        X1, Y1, X2, Y2 = int((2*X0 + X3)/3), int((2*Y0 + Y3)/3), int((X0 + 2*X3)/3), int((Y0 + 2*Y3)/3)

                
        while True:
            _, capimg = cap.read()
            disp_img = np.array(capimg)

            cv2.rectangle(disp_img, (X0, Y0), (X3, Y3), ROI_RECT_COLOUR, 5)

            roi = capimg[Y0:Y3, X0:X3]

            colors = read_colors_from_roi(roi)

            print colors, '\n\n\n\n'
            
            cv2.imshow('dsp', disp_img)
            if cv2.waitKey(1) == ord('q'):
                break

            
        
    finally:
        cv2.destroyAllWindows()
        cap.release()
    

DEBUG = True

#white green blue red yellow orange
#white | red orange yellow green blue

if DEBUG:
    colors = np.array([[57, 3, 98], [112, 94, 70],
                  [212, 82, 70], [1, 68, 78],
                  [56, 81, 91], [39, 80, 91]]*9)

print classify(colors)
'''
