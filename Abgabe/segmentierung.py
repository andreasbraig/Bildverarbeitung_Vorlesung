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


def marker(image,segm):

    #segm[segm != (5, 5, 5) or segm != (6, 6, 6) or segm != (7, 7, 7)] = 0 
    #copy = np.zeros_like(segm)
    #for i, row in enumerate(segm):
    #    for j, col in enumerate(row):
    #        if col[0] == 5 or col[0] == 6 or col[0] == 7:
    #            copy[i, j] = (255, 255, 255)

    

    #segm = global_kontrastspeizung(segm)
     
    segm_gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)

    mask = (segm_gray == 5) | (segm_gray == 6) | (segm_gray == 7)

    image_binary = np.zeros_like(segm_gray)

    image_binary[mask] = 255

    _, segm_binary = cv2.threshold(segm_gray,cf["val1"],cf["val2"],cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image_binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)

    for idx, cont in enumerate(contours[:len(contours)]):
        # Berechne die Momente der Kontur, um den Mittelpunkt zu finden
        M = cv2.moments(cont)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.drawMarker(image,(cx,cy),(255,255,255))

    segm_binary= cv2.cvtColor(image_binary, cv2.COLOR_GRAY2BGR)

    return np.hstack([image,segm_binary])






def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#show_image("test",marker(image,segm))

def run(image,image2, result,settings=None): #Funktion zur Bildverarbeitung

    result.append({"name":"Gray","data":marker(image,image2)})


if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    print(result)
    run(image,image2,result)   
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()