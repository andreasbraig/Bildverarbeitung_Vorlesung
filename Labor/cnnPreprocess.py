import numpy as np 
import os
import cv2
import random
import shutil

import labor6 as l6

#Kopiert einen Ordner zu einem Ordner!
def preprocess_folder(source,dst):

    #Prüfen ob dst existiert, wenn nicht erstellen
    if not os.path.exists(dst):
        os.makedirs(dst)

    #print(os.listdir(source))

    #alle elemente im Source Pfad bekomme
    for element in os.listdir(source):
        #alle verarbeiten
        image=cv2.imread(os.path.join(source, element))
        
        output_path = os.path.join(dst, element)
        
        #Bildvearbeitung
        result=l6.scale_img(image)

        #Speichern in neuem Ordner
        cv2.imwrite(output_path,result)

def train_test_split(source,dst):

    if not os.path.exists(dst):
        os.makedirs(dst)

    files = [f for f in os.listdir(source)]

    random_samples = random.sample(files, int(len(files)*0.1))

    for file in random_samples:
        shutil.move(os.path.join(source,file),os.path.join(dst,file))

def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print("Jibbet Nich")


path = "../Bilder/Cashews"

cleanup(path+"/Lernen")
cleanup(path+"/Test")

preprocess_folder(source= path+"/unverarbeitet/Anomaly",
                  dst=path+"/Lernen/Anomaly")
print("Lernen Anomaly Fertig")
preprocess_folder(source=path+"/unverarbeitet/Normal",
                  dst=path+"/Lernen/Normal")
print("Lernen Normal Fertig")


train_test_split(source=path+"/Lernen/Anomaly",
                 dst=path+"/Test/Anomaly")
train_test_split(source=path+"/Lernen/Normal",
                 dst=path+"/Test/Normal")

