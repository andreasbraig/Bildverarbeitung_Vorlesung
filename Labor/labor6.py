import numpy as np 
import os
import cv2

def getcontours(image,settings=(10,50)):
  svm = cv2.ml.SVM_create()
  svm.setKernel(cv2.ml.SVM_LINEAR)
  #svm.setKernel(cv2.ml.SVM_RBF)
  svm.setType(cv2.ml.SVM_C_SVC)
  svm.setC(settings[0]/10+0.1)
  svm.setGamma(settings[0]/10.0+0.1)

  h,w=image.shape[0:2]

  #Definition der Farben und klassen immer oberer und unterer Wert 

  rows=np.zeros((12,4))
  #Cashew Farbe
  rows[0,:]=[127,185,234,1]
  rows[1,:]=[29,87,146,1]
  rows[2,:]=[69,151,209,1]
  rows[3,:]=[94,167,219,1]
  rows[4,:]=[109,188,237,1]
  rows[5,:]=[81,165,223,1]


  #Hintergrund?
  rows[6,:]=[5, 13, 12,0]
  rows[7,:]=[16, 24, 24,0]
  rows[8,:]=[1, 5, 6,0]
  rows[9,:]=[8, 15, 18,0]
  rows[10,:]=[53, 62, 75,0]
  rows[11,:]=[12, 22, 29,0]

  #Training -> Dimensionen jetzt auf 3 erhöht 
  train = rows[:,0:3].astype(np.float32) 
  response= rows[:,3].astype(int) 

  svm.train(train, cv2.ml.ROW_SAMPLE, response)
  
  #print(image.shape)
  mat=image.reshape(-1,3)
  #print(mat.shape)
         
  #Gebe dem ganzen jeden Pixel und lasse ihn Predicten was er ist 
  erg = svm.predict(mat.astype(np.float32))
  erg = erg[1].reshape(h,w)
  #erweiterung um Color Result 
  color_result = np.uint8(train[np.int32(2*erg),:])
  #print(color_result.shape)

  cv2.normalize(erg,erg,0, 1, cv2.NORM_MINMAX)
  #erg=cv2.resize(erg,None,None,10,10,cv2.INTER_NEAREST)

  erg = np.uint8(erg*255)

  return erg

def isolate_largest_contour(blob,image):



    contours, hierarchy = cv2.findContours(blob,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    largest_contour = max(contours, key=cv2.contourArea) if contours else None


    rect = cv2.minAreaRect(largest_contour)
    #print(rect[2])

    box = cv2.boxPoints(rect)
    box = np.intp(box)  # In ganze Zahlen umwandeln
    #print(box)


    # Wähle drei Punkte für die Affine Transformation (z.B. die oberen drei)
    #Je nach Rotation der Nuss ist der erkannte Winkel in Rect (rect[2]) anders und die Reihenfolge der Punkte ist nicht mehr 
    #OL,UL,UR,OR sondern bei winkeln kleiner 45: OR,OL,UL,UR Daher muss die berechnung von height und width angepasst werden und auch die Source Points 
    if rect[2] > 45:
        src_pts = np.float32([box[0], box[1], box[2]])
        width = int(rect[1][1])
        height = int(rect[1][0])
    else:
        src_pts = np.float32([box[1], box[2], box[3]])
        width = int(rect[1][0])
        height = int(rect[1][1])


    # Zielpunkte: Diese bestimmen das "gerade" Endergebnis
    dst_pts = np.float32([[0, 0], [width, 0], [width, height]])

    # Berechnung der Affinen Transformationsmatrix
    M = cv2.getAffineTransform(src_pts, dst_pts)

    # Anwenden der Affinen Transformation
    cropped = cv2.warpAffine(image, M, (width, height))
    cropped_binary = cv2.warpAffine(blob, M, (width, height))

    #Konturanalyse des zugeschnittenen Bildes
    contours,hierarchy = cv2.findContours(cropped_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    #mask als bild so initialisieren, dass es gleich groß ist
    mask=np.zeros_like(cropped)
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY) #Konversion zu einkanal Graubild


    #Konturmmaske zeichnen
    cv2.drawContours(contours=[largest_contour], 
                     color=255,
                     hierarchy=hierarchy,
                     image=mask,
                     contourIdx=0,
                     lineType=cv2.LINE_8,
                     maxLevel=100,
                     thickness= -1)

    #maske mit Bild "verheiraten"    

    result = cv2.bitwise_and(cropped,cropped, mask=mask)

    return result

def scale_img(image,dim=150):

    #isoliere das zielbild
    isolated = isolate_largest_contour(getcontours(image),image)

    # Skalierung
    h, w = isolated.shape[0:2]
    #Leeres Quadratisches Bild erstellen
    result = np.zeros((dim,dim,3),np.uint8)
    #Berechnung der Skalierungsfaktoren für das isolierte Bild
    scale = dim / max(h, w)
    breite = int(w * scale)
    hoehe = int(h * scale)
    dimensionen = (breite, hoehe)
    scaled = cv2.resize(isolated,dimensionen) #Tatsächliche skalierung

    #Schreibe das isolierte Bild in das leere
    
    result[:scaled.shape[0], :scaled.shape[1]] = scaled

    return result

#__________________AUSFÜHRENDER BEREICH___________________

def run(image, result,settings=None): #Funktion zur Bildverarbeitung
    newimage = scale_img(image)



    #Graubild erzeugen
    image3=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    result.append({"name":"Gray","data":newimage})

if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]



    #10% Verschieben
    
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()