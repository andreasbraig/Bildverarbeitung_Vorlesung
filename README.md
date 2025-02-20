# README der Gewinnertruppe Bildverarbeitung

---

## Kurzbeschreibung

Dieses Repo wird für die Zusammenarbeit/ gemeinsame Bearbeitung der Laboraufgaben und der Abgabe verwendet

## Grundlagen

### Venv erstellen:

Sinn ist, bugs zu vermeiden und wir können bei Komplikationen mit Heintz auch die requirements.txt übergeben. 

1. Venv im Verzeichnis erstellen

   1. Ins Hauptverzeichnis des Repos navigieren
   2. python3.11 -m venv .venv
   3. Der Ordner wird von der gitignore ausgeschlossen
2. venv Aktivieren:

   1. Mac: source .venv/bin/activate
   2. Windows: cd .venv/bin und dann ./Activate.ps1
3. Alle Requirements aus dem Repo übernehmen:

   1. pip install -r /path/to/requirements.txt
4. Wenn man das Venv deaktivieren möchte:

   1. deactivate

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

    let rgb = new cv.MatVector(); <- das is doch Java Script oder bin ich Dumm? (Andi)
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
