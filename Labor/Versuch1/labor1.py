import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy

image=cv2.imread('../../Bilder/sample1.jpg') #Bild laden


#Utilities 
def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Aufgaben:_____________________

def mirror(image):

    #Größenbestimmung
    h, w = image.shape[0:2]

    #iteriere über jede Zeile
    for height in image:
        #für alle spaleten bis zu hälfte
        for i in range(int(w/2)):

            #Tausche Die letzte mit der ersten [1,2,...,w-1,w] -> [w,w-1,...,2,1] und das für jede Zeile
            height[i], height[w-1-i] = height[w-1-i].copy(), height[i].copy()

    return image


def invert_grey(image):

    #Emptylike erstellt ein leeres Array mit gleicher Shape (Zeilen Spalten....) wie vorgabearray
    #Gottlos geiler Hilfebefehl man
    newimage = np.empty_like(image)
    #newimage2 = np.empty_like(image) zum testen ob nur graukonvertieren geht

    #Pixelweise operation
    for i,height in enumerate(image):
        for j,width in enumerate(height):
                #nehme jeden Farbvektor, bilde den Mittelwert und ziehe diesen von 255 ab
                newimage[i][j] = 255-width.mean() 
                #newimage2[i][j] = width.mean() 
            
    #ANMERKUNG: Keine korrekte Schwarzweiß konvertierung, da normalerweise der Grüne Farbkanal stärker gewichtet wird.
    #Menschliche Augen nehmen Grün besser wahr. 

    return newimage

def invert_color(image):

    # Muss ich das jetzt nochmal kommentieren???

    newimage=np.empty_like(image)

    for i,height in enumerate(image):
        for j,width in enumerate(height):
            for k,color in enumerate(width):
                newimage[i][j][k] = 255-color

    return newimage

#AUSFÜHRUNG:____________________

image = invert_grey(image)
show_image("test",image)

