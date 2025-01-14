import cv2 #Importieren OpenCV
import numpy as np #Importieren Numpy

image=cv2.imread('../Bilder/sample1.jpg') #Bild laden
w, h = image.shape[0:2]


x = int(input("X Wert: "))
y = int(input("Y Wert: "))

print(x, y)

value = image[x, w, 0]

print(type(value))

print("BGR Werte: ", image[x, w])