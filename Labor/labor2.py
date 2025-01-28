import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy
from matplotlib import pyplot as plt


def global_kontrastspeizung(image):

    #bestimme die maximale Dehnung des vorhandenen Bildes
    maxpos = np.max(image.copy())
    minpos = np.min(image.copy())


    f = (255/(maxpos-minpos))

    #berechne nach Formel aus Vorlesung
    img  = np.uint8(((image - minpos) * (f)))

    return img



def kanal_kontrastspeizung(image):

    #teile das Bild in Schwarz weiß fotos der einzelnen Kanäle
    b,g,r =cv2.split(image)

    #merge und führe Globale spreizung für jedes Foto durch 
    result = cv2.merge([global_kontrastspeizung(b),global_kontrastspeizung(g),global_kontrastspeizung(r)])

    return result

def pseudokolorierung(image):


    b = np.uint8(np.sin((2*np.pi/255)*image)*255)
    g = np.uint8(np.cos((2*np.pi/255)*image)* -1 *255)
    r = np.uint8(np.sin((2*np.pi/255)*image)* -1 *255)

    newimage = cv2.merge([b,g,r])

    return newimage

def whitebalance(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    result = cv2.minMaxLoc(gray)

    print(result[3][0])

    b,g,r = cv2.split(image)

    b = b * (255/image[result[3][0]][result[3][1]])[0]
    g = g * (255/image[result[3][0]][result[3][1]])[0]
    r = r * (255/image[result[3][0]][result[3][1]])[0]

    newimage = cv2.merge(np.uint8([b,g,r]))

    return newimage


#Utilities 
def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run(image, result,settings=None): #Funktion zur Bildverarbeitung
    #Graubild erzeugen
    image3=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result.append({"name":"Gray","data":kanal_kontrastspeizung(image)})


if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    run(image,result)
    
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()