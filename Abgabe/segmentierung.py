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

    image_binary = np.zeros_like(segm_gray)

    image_binary[mask] = 255

    contours, _ = cv2.findContours(image_binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    #print(len(contours))
    for idx, cont in enumerate(contours[:len(contours)]):
        # Berechne die Momente der Kontur, um den Mittelpunkt zu finden
        M = cv2.moments(cont)

        if len(cont)>10:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            corner.append([cx, cy])
            #print("neue Contur:",len(cont))
        else:
            pass

    return  corner, image_binary


def marker(image, segm):

    corner, binary = eye_mouth(segm)

    for cor in corner:
        cv2.drawMarker(image,cor,(255,255,255))

    seg = global_kontrastspeizung(segm)

    return image, binary, seg
    






def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#show_image("test",marker(image,segm))

def run(image,image2, result,settings=None): #Funktion zur Bildverarbeitung
    mark, bin, seg = marker(image, image2)
    # seg = global_kontrastspeizung(image2)
    # _,bin = eye_mouth(image2)
    result.append({"name":"res","data":mark})
    result.append({"name":"binary","data":bin})
    result.append({"name":"spreiz","data":seg})
    



if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    print(result)
    run(image, image2, result)   
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()