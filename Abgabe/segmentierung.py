import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy
from matplotlib import pyplot as plt
import json
import math

cf = json.load(open("ImageProcessingGUI.json", 'r'))

#image_path = "Datensatz/Images/4961fdd8-5b8c-428e-bcd1-8f65f8aecfe0.jpg"
#sem_seg = "Datensatz/Images/4961fdd8-5b8c-428e-bcd1-8f65f8aecfe0.png"

#image=cv2.imread(image_path) #Bild laden (im unterordner)
#segm = cv2.imread(sem_seg) #Bild laden (im unterordner)

def global_kontrastspeizung(image):

    #bestimme die maximale Dehnung des vorhandenen Bildes
    maxpos = np.max(image.copy())
    minpos = np.min(image.copy())


    f = (255/(maxpos-minpos))

    #berechne nach Formel aus Vorlesung
    img  = np.uint8(((image - minpos) * (f)))

    return img


def freistellen(image, segm):

    return image, segm


def transformation(image, triangle):
    
    return


def eye_mouth(segm):

    corner = []

    segm_gray = cv2.cvtColor(segm, cv2.COLOR_BGR2GRAY)

    mask = (segm_gray == 5) | (segm_gray == 6) | (segm_gray == 7)

    image_binary = np.zeros_like(segm_gray)

    image_binary[mask] = 255

    contours, _ = cv2.findContours(image_binary,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    #print(len(contours))


    for idx, cont in enumerate(contours[:min(4, len(contours))]):
        # Berechne die Momente der Kontur, um den Mittelpunkt zu finden
        M = cv2.moments(cont)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        corner.append([cx, cy])
        #print("neue Contur:",len(cont))
     


    # corner in Augen und Mund unterteilen
    eyes, mouth = eye_mouth_split(corner)
    # Verhindern, dass zu viele Marker eng beieinander gesetzt werden, Mittelung des größten Clusters
    if len(mouth) > 1:
        mouth = [cluster_points(mouth, eyes)]  # Clustert Punkte, die zu nah aneinander liegen (abhängig vom Abstand der Augen)
    corner = eyes + mouth
    return  corner, image_binary


def marker(image, segm):

    corner, binary = eye_mouth(segm)

    for cor in corner:
        cv2.drawMarker(image,cor,(255,255,255))

    seg = global_kontrastspeizung(segm)

    return image, binary, seg
    
def eye_mouth_split(corner):
    if len(corner) < 3:
        return corner, []  # Falls zu wenige Punkte vorhanden sind, alle zu "eyes" packen

    corner.sort(key=lambda p: p[1])  # Sortiere nach der y-Koordinate
    eyes = corner[:2]  # Die zwei obersten Punkte sind die Augen
    mouth = corner[2:]  # Die restlichen Punkte sind der Mund

    return eyes, mouth

def calculate_eye_distance(eyes):
    if len(eyes) < 2:
        return 0  # Falls weniger als 2 Augen vorhanden sind, gebe 0 zurück
    
    # Berechne den Abstand zwischen den beiden Augen
    x1, y1 = eyes[0]
    x2, y2 = eyes[1]
    
    eye_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # print(dist)
    
    return eye_distance

def cluster_points(points, eyes):
    clustered = []
    max_distance = calculate_eye_distance(eyes) * 0.01

    # Wir gehen alle Punkte durch
    for pnt in points:
        # Falls der Cluster leer ist, füge den Punkt direkt hinzu
        if not clustered:
            clustered.append([pnt])
            continue
        
        # Berechne die Distanz zum Mittelpunkt jedes bestehenden Clusters
        added_to_cluster = False
        for cluster in clustered:
            cluster_center = np.mean(cluster, axis=0)  # Berechne den Mittelpunkt des Clusters
            dist = np.linalg.norm(np.array(pnt) - np.array(cluster_center))  # Euklidische Distanz
            
            if dist < max_distance:
                cluster.append(pnt)  # Füge den Punkt zum Cluster hinzu
                added_to_cluster = True
                break
        
        # Wenn der Punkt keinem Cluster hinzugefügt wurde, erstelle einen neuen Cluster
        if not added_to_cluster:
            clustered.append([pnt])

    # Finde das Cluster mit der maximalen Anzahl von Punkten
    largest_cluster = max(clustered, key=len)  # Cluster mit den meisten Punkten
    averaged_largest_cluster = np.mean(largest_cluster, axis=0)  # Mittelwert des größten Clusters
    averaged_largest_cluster = np.round(averaged_largest_cluster).astype(int).tolist()  # Runden und zu int konvertieren

    return averaged_largest_cluster



def show_image(text, image):
    cv2.imshow(text, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#show_image("test",marker(image,segm))

def run(image,image2, result,settings=None): #Funktion zur Bildverarbeitung
    mark, bin, seg = marker(image, image2)
    # seg = global_kontrastspeizung(image2)
    # _,bin = eye_mouth(image2)
    result.append({"name":"res","data":mark})
    result.append({"name":"binary","data":bin})
    result.append({"name":"spreiz","data":seg})
    



if __name__ == '__main__': #Wird das Skript mit python Basis.py aufgerufen, ist diese Bedingung erfüllt
    image=cv2.imread("Images\Ball.jpg")
    result=[]
    print(result)
    run(image, image2, result)   
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()