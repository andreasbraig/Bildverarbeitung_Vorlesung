import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy

def freistellen(image, segm):

    segm_gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)
    image_binary = np.zeros_like(segm_gray)

    mask = (segm_gray != 0)

    image_binary[mask] = 255

    result =  cv2.bitwise_and(image,image, mask=image_binary)
    
    b,g,r = cv2.split(result)

    rgba_image = cv2.merge([b, g, r, image_binary])

    return rgba_image


def transformation(image, segm,eye_dist=70):
    centr = get_corners(segm)
   

    target_pts = np.float32([
        [eye_dist*3//2, eye_dist*3], 
        [eye_dist*2, eye_dist*2], 
        [eye_dist, eye_dist*2]
    ])
    src_pts = np.float32([centr[2], centr[1], centr[0]])
    
    # Transformation berechnen
    matrix = cv2.getAffineTransform(src_pts, target_pts)
    warped = cv2.warpAffine(image, matrix, (eye_dist*3, eye_dist*4))
    
    # Bild speichern
    return warped

def get_lastpoint(centr):

    vec = np.subtract(centr[1] , centr[0])
    vec2 = [vec[1], vec[0]]

    last = np.add(centr[0], vec//2)
    last = np.add(last, vec2)
    return last

def get_single_center(segm,value):

    segm_gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)

    mask = (segm_gray == value)

    image_binary = np.zeros_like(segm_gray,dtype=np.uint8)

    image_binary[mask] = 255

    contours, _ = cv2.findContours(image_binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

    if len(contours)==0:
        return 0
    else: 

        largest = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return [cx,cy]

def get_corners(segm):
    result=[]
    contours = [5,6,7]

    for ele in contours:
        coordinates = get_single_center(segm,ele)
        if coordinates != 0:
            result.append(coordinates)

    if len(result) != 3:       
        result.append(get_lastpoint(result))

    return result

if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    pass