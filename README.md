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

### Umwandlung in Graubild
    cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);

### Minimaler und maximaler Wert
    let res = cv.minMaxLoc(dst); 
        res.minVal
        res.maxVal
        res.minLoc
        res.maxLoc

### Bildkanäle extrahieren
    let rgb = new cv.MatVector();
    cv.split(src, rgb);

### Erstellung einer neuen Matrix der Größe einer anderen mit initialVal
    initialVal = 1
    dstX = new cv.Mat(src.rows, src.cols, cv.CV_8U, new cv.Scalar(initialVal));

### Multiplikation aller Werte eines Channels mit einem Faktor
    factor = 1
    channel = 0
    channelMatrix = dstX.mul(rgb.get(channel), factor);

### Aktualisierung (Überschreibung) eines Channels
    destChannel = 0
    rgb.set(destChannel, channel);

### Zusammenführen der Channels zu einer Matrix
    cv.merge(rgb, dst);
