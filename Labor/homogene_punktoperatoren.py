import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy

image=cv2.imread('../Bilder/Ball.jpg') #Bild laden
print(image.shape) #Zeigt Informationen über die Größe und Dimension der Bildmatrix
h,w=image.shape[0:2] #Größen und Kanalanzahl aus Shape auslesen

resImage1 =np.zeros((h,w),np.uint8)#Leeres 8-Bit Graubilder erzeugen

#Punktoperation auf Pixelbasis
for y in range(image.shape[0]):
  for x in range(image.shape[1]):
    resImage1[y,x]=np.uint8(image[y,x,1]/2+5)

#Punktoperation auf Matrix
resImage2 = np.uint8(image/2+5)

cv2.imshow("Pixel",resImage1)
cv2.imshow("Matrix",resImage2)
cv2.waitKey(0)
cv2.destroyAllWindows()