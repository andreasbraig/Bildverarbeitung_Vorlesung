import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy

image=cv2.imread('../../Bilder/sample1_lowcontrast.jpg') #Bild laden (im unterordner)


def kontrastspreizung(image):

    #Kontrastanalyse 



    #kontrastberechnung

    for height in image:

        newimage = np.matrix(height)
        print(newimage)


    #zu Bild zusammenführen

    

    return newimage


#Utilities 
def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


kontrastspreizung(image)