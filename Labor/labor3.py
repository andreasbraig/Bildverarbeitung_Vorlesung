import cv2
import json
import numpy as np
import math

#
def segmentierung(image):
    cf = json.load(open("ImageProcessingGUI.json", 'r'))
    h, w = image.shape[:2]
    image2 = np.zeros((h, w, 3), np.uint8)
    
    # Konvertiere das Bild in den HSV-Farbraum für eine bessere Segmentierung
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definiere Farbgrenzen für die Segmentierung
    color_ranges = {
        "red1": ((0, 100, 100), (10, 255, 255)),
        "red2": ((0, 0 ,0 ), (20, 255, 255)),
        "green": ((40, 40, 40), (90, 255, 255)),
        "blue": ((100, 100, 40), (140, 255, 255)),
        "yellow": ((20, 100, 100), (40, 255, 255)),
        "orange": ((10, 100, 100), (25, 255, 255))
    }
    
    # Färbe die jeweiligen Segmente
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(image_hsv, np.array(lower), np.array(upper))
        if color == "red1":
            image2[mask > 0] = (0, 0, 255)
        elif color == "red2":
            image2[mask> 0] = (0,0,128)
        elif color == "green":
            image2[mask > 0] = (0, 255, 0)
        elif color == "blue":
            image2[mask > 0] = (255, 0, 0)
        elif color == "yellow":
            image2[mask > 0] = (0, 255, 255)
        elif color == "orange":
            image2[mask > 0] = (0, 165, 255)
    
    #Unvollständig aber Konzept erfüllt: Laut Heintz -> Jetzt nur noch optimierung aber kein wissensgewinn mehr

    return image_hsv

#Also es geht darum ein Bild zu segmentieren, dann konturen zu zeichnen und die dann zu füllen 
def blobanalyse(image):

    #hilfsfunktion um die Slider in den code zu bekommen, nicht mehr genutzt. 
    cf = json.load(open("ImageProcessingGUI.json", 'r'))

    #erzeugung eines leren uint8 bildes
    h,w = image.shape[0:2]
    newimage = np.uint8(np.zeros((h,w,3)))

    #graukonversion damit die Konturen mit high und low eingegrenzt werden können
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray,105,177,cv2.THRESH_BINARY_INV)

    #automatische konturerkennung und sammlung der Daten
    contours, hierarchy = cv2.findContours(image_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #über die gefundenen Konturen iterieren und dann die bereiche füllen
    for i,cont in enumerate(contours):
        
        b = np.round(np.random.default_rng().random()*255)
        g = np.round(np.random.default_rng().random()*255)
        r = np.round(np.random.default_rng().random()*255)

        cv2.drawContours(contours=contours,
                         color=(int(b),int(g),int(r)),
                         hierarchy=hierarchy,
                         image=newimage,
                         contourIdx=i,
                         lineType=cv2.LINE_8,
                         maxLevel=100,
                        thickness= -1)

    return newimage


#AUSFÜHRBARER BEREICH



def run(image, result,settings=None): #Funktion zur Bildverarbeitung
    #Graubild erzeugen
    image3=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result.append({"name":"Gray","data":blobanalyse(image)}) #hier funktion eijnfügen, doe getestet werden soll


if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    run(image,result)
    
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()