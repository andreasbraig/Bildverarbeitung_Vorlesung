import cv2

import Labor.Versuch2.labor2 as l2

def run(image, result,settings=None): #Funktion zur Bildverarbeitung
    #Graubild erzeugen
    image3=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result.append({"name":"Gray","data":l2.kanal_kontrastspeizung(image)})


if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    run(image,result)
    
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()