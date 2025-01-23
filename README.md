# README der Gewinnertruppe Bildverarbeitung
---


## Kurzbeschreibung
Dieses Repo wird für die Zusammenarbeit/ gemeinsame Bearbeitung der Laboraufgaben und der Abgabe verwendet

## Grundlagen

### Template:
    import cv2 #Importieren OpenCV
    import numpy as np #Importieren Numpy

    image=cv2.imread('../../Bilder/sample1.jpg') #Bild laden (im unterordner)


    #Utilities 
    def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

### Matrix-Berechnung
    np.Matrix * Skalar
