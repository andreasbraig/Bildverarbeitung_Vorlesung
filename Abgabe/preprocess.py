import cv2 
import numpy as np 
from matplotlib import pyplot as plt
import json
import os

import segmentierung as sg

def create_gender_array(data):

    array = []  # Initialisiere ein leeres 2D-Array

    for info in data:
        name = info.get("name", "Unknown")  # Falls "name" fehlt, setze "Unknown"
        gender = info.get("gender", "Unknown")  # Falls "gender" fehlt, setze "Unknown"
        array.append([name, gender])  # Füge als neue Zeile hinzu

    return array

def split_genders(dst_m,dst_w,gender_array):

    if not os.path.exists(dst_m):
        os.makedirs(dst_m)
        
    if not os.path.exists(dst_w):
        os.makedirs(dst_w)
    
    data_suffix = [".jpg",".png"]

    for element in gender_array:
        #print(element[0][-4:])
        path = element[0]
        filename = path[7:-4]
        if path[-4:] == data_suffix[0]:
            
            image=cv2.imread(path)
            segmentpath = path[:-4]+data_suffix[1]
            segm = cv2.imread(segmentpath)

            free,_ =sg.freistellen(image,segm)

            if element[1] == "M":
                print("Success")
                cv2.imwrite(os.path.join(dst_m, filename+data_suffix[1]),free)
            elif element[1] == "W":
                cv2.imwrite(os.path.join(dst_w, filename+data_suffix[1]),free)
            else:
                print("Fehler, Kein eindeutiges Gender")


    #for element in os.listdir(source):
 
    





#____________________Ausführung________________

data = json.load(open("Images/tag.json","r"))

split_genders(dst_m= "Datensatz/maennlich",
              dst_w= "Datensatz/weiblich",
              gender_array=create_gender_array(data))