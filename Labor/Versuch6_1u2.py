import random
import shutil
import cv2
import numpy as np
import os

def run(image, result,settings=None):
    image=image.copy()
    image2=image.copy()
    r, g, b = cv2.split(image)

    image2[ (r >g) | (g<25) | ((g*1.5 >b) & (b<140))] = (0,0,0)

    _, thresh = cv2.threshold(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY),10,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i,ele in enumerate(contours):
        if cv2.contourArea(ele)>30000:
            rect = cv2.minAreaRect(ele)
            
            #rect anpassen, damit immer Querformat
            if(rect[1][1]>rect[1][0]):
                rect=[rect[0],[rect[1][1],rect[1][0]],rect[2]-90]
            
            boxf = cv2.boxPoints(rect)
            cv2.drawContours(image,[np.int0(boxf)],0,(0,0,255),4)

            #Hintergrund ausblenden
            mask=np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
            cv2.drawContours(mask, [ele], -1, (1,1,1),cv2.FILLED,8)
            image2[mask==0]=(0,0,0)


            #Mittels 4 Punkt Transformation auf Rechteck mit 600x400 abbilden. Eigentlich ist zu erwarten, dass 0 und size[1]-1 zu verwenden ist. Der Bereich muss aber um 1 angepasst werden, daher -1 und size[1].Alternativ kann das Bild auch gedreht und dann beschnitten werden. Das Beschneiden ist aber nicht so einfach. Beschneiden durch erneutes findContours ist möglich aber unschön da rechenintensiv.
            size=(600,400)
            dst_pts = np.array([[-1, size[1]],
                        [-1, -1],
                        [size[0], -1],
                        [size[0], size[1]]], dtype="float32")
            M = cv2.getPerspectiveTransform(boxf.astype("float32"), dst_pts)
            #bug: warpPerspective verwendet (in meiner Version) immer Nearest Neigthbor daher wird erst auf 600,400 gewarpt und dann auf die Zielgröße von 150,100 interpoliert
            warped = cv2.warpPerspective(image2, M, (size[0], size[1]),cv2.INTER_CUBIC)
            warped = cv2.resize(warped, (150,100), interpolation= cv2.INTER_CUBIC)

            result.append({"name":"Location","data":image})
            result.append({"name":"Segmentation","data":image2})
            result.append({"name":"Result","data":warped})

if __name__ == '__main__':
    baseDir="cashew\Data"
    os.makedirs(baseDir+"\Results", exist_ok=True)
    for path, subdirs, files in os.walk(baseDir+"\Images"):
        files = [ fi for fi in files if fi.endswith(".JPG") ]
        className=path.split(os.sep)[-1]
        
        if(len(files)>0):
            os.makedirs(baseDir+"\Results\\Train\\"+className, exist_ok=True)
            
        for name in files:
            fileName=name
            print(className," ",name)
            fileNameFull=os.path.join(path, name)
            image=cv2.imread(fileNameFull)
            result=[]
            run(image,result)
            for ele in result:
                if(ele["name"]=="Result"):
                    cv2.imwrite(baseDir+"\Results\\Train\\"+className+"\\"+name,ele["data"])
    
    for path, subdirs, files in os.walk(baseDir+"\Results\\Train\\"):
        files = [ fi for fi in files if fi.endswith(".JPG") ]
        className=path.split(os.sep)[-1]

        if(len(files)>0):
            os.makedirs(baseDir+"\Results\\Test\\"+className, exist_ok=True)

        count=int(len(files)*0.1)
        for i in range(count):
            index=int(random.random()*len(files))
            shutil.move(os.path.join(path, files[index]),baseDir+"\Results\\Test\\"+className+"\\"+files[index])
            files.remove(files[index])
                
    cv2.waitKey(0)
    cv2.destroyAllWindows()