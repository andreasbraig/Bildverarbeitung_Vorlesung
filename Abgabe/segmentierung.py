import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy
from matplotlib import pyplot as plt
import json


cf = json.load(open("ImageProcessingGUI.json", 'r'))

#image_path = "Datensatz/Images/4961fdd8-5b8c-428e-bcd1-8f65f8aecfe0.jpg"
#sem_seg = "Datensatz/Images/4961fdd8-5b8c-428e-bcd1-8f65f8aecfe0.png"

#image=cv2.imread(image_path) #Bild laden (im unterordner)
#segm = cv2.imread(sem_seg) #Bild laden (im unterordner)

def global_kontrastspeizung(image):

    #bestimme die maximale Dehnung des vorhandenen Bildes
    maxpos = np.max(image.copy())
    minpos = np.min(image.copy())


    f = (255/(maxpos-minpos))

    #berechne nach Formel aus Vorlesung
    img  = np.uint8(((image - minpos) * (f)))

    return img


def freistellen(image, segm):

    return image, segm


def transformation(image, triangle):
    
    return


def eye_mouth(segm):

    corner = []

    segm_gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)

    mask = (segm_gray == 5) | (segm_gray == 6) | (segm_gray == 7)

    image_binary = np.zeros_like(segm_gray,dtype=np.uint8)

    image_binary[mask] = 255

    contours, _ = cv2.findContours(image_binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    for idx, cont in enumerate(contours[:len(contours)]):
        # Berechne die Momente der Kontur, um den Mittelpunkt zu finden
        M = cv2.moments(cont)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        corner.append([cx, cy])

    return corner, image_binary


def marker(image, segm):

    corner, binary = eye_mouth(segm)

    for cor in corner[2:]:
        cv2.drawMarker(image,cor,(255,255,255))

    seg = global_kontrastspeizung(segm)

    return image, binary, seg
    


def segment_filter(image, segm):
    centroids, _ = eye_mouth(segm)

    if len(centroids) != 3:
        raise ValueError("Nicht genau drei Segmentschwerpunkte gefunden!")

    # target_pts = np.float32([mouth, eyes[0], eyes[1]])

    eye_dist = 70

    target_pts = np.float32([
        [eye_dist*3//2, eye_dist*3], 
        [eye_dist*2, eye_dist*2], 
        [eye_dist, eye_dist*2]
    ])
    src_pts = np.float32([centroids[0], centroids[1], centroids[2]])
    
    # Transformation berechnen
    matrix = cv2.getAffineTransform(src_pts, target_pts)
    warped = cv2.warpAffine(image, matrix, (eye_dist*3, eye_dist*4))
    
    # Bild speichern
    return warped



def run(image,image2, result,settings=None): #Funktion zur Bildverarbeitung
    seg = segment_filter(image, image2)
    result.append({"name":"cropped","data":seg})
    return
    



if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    
    result=[]
    print(result)
    run(image, image2, result)   
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()