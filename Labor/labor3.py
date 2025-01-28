import cv2
import numpy as np

def Segmentierung(image):

    newimage = np.empty_like(image)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, image_binary = cv2.threshold(image_gray,130,1,cv2.THRESH_BINARY_INV)

    for i,height in enumerate(image):
        for j, width in enumerate(height):
            j



    return image_binary


#AUSFÜHRBARER BEREICH

def run(image, result,settings=None): #Funktion zur Bildverarbeitung
    #Graubild erzeugen
    image3=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result.append({"name":"Gray","data":Segmentierung(image)})


if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    run(image,result)
    
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()