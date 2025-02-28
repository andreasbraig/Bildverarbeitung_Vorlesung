import cv2 
import json
import os
import shutil
import random

import preprocess as pp

def create_gender_array(data):

    array = []

    for info in data:
        name = info.get("name")
        gender = info.get("gender")
        array.append(["Images/"+name[7:], gender])

    return array


def split_genders(dst_m,dst_w,gender_array,eye_dist=70):

    if not os.path.exists(dst_m):
        os.makedirs(dst_m)
        
    if not os.path.exists(dst_w):
        os.makedirs(dst_w)
    
    data_suffix = [".jpg",".png"]

    for element in gender_array:

        path = element[0]
        filename = path[7:-4]
        if path[-4:] == data_suffix[0]:
            
            image=cv2.imread(path)
            segmentpath = path[:-4]+data_suffix[1]
            segm = cv2.imread(segmentpath)

            free = pp.freistellen(image,segm)
            warped = pp.transformation(free,segm,eye_dist)

            if element[1] == "M":
                cv2.imwrite(os.path.join(dst_m, filename+data_suffix[1]),warped)
            elif element[1] == "W":
                cv2.imwrite(os.path.join(dst_w, filename+data_suffix[1]),warped)
            else:
                print("Fehler, Kein eindeutiges Gender")
    

def train_test_split(source,dst,ratio=20):

    if not os.path.exists(dst):
        os.makedirs(dst)

    files = [f for f in os.listdir(source)]

    random_samples = random.sample(files, int(len(files)*(ratio/100)))

    for file in random_samples:
        shutil.move(os.path.join(source,file),os.path.join(dst,file))

 
def cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print("Jibbet Nich")


def preprocess(eye_dist):

    data = json.load(open("Images/tag.json","r"))
    
    cleanup("Datensatz/Learn")
    cleanup("Datensatz/Test")
    
    all_m = "Datensatz/Learn/maennlich"
    all_w = "Datensatz/Learn/weiblich"
    
    split_genders(dst_m= all_m, 
                  dst_w= all_w ,
                  gender_array=create_gender_array(data),
                  eye_dist=eye_dist)
    
    print("Transformation abgeschlossen")
    
    train_test_split(source=all_m, 
                     dst="Datensatz/Test/maennlich", 
                     ratio=20)
    train_test_split(source=all_w, 
                     dst="Datensatz/Test/weiblich", 
                     ratio=20)
    
    print("Train_Test_Split abgeschlossen")

if __name__ == "__main__":

    preprocess(eye_dist=100)