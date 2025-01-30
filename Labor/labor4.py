import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

def random_colorgen():

    b = np.round(np.random.default_rng().random()*255)
    g = np.round(np.random.default_rng().random()*255)
    r = np.round(np.random.default_rng().random()*255)

    return (int(b),int(g),int(r))

def konturanalyse(image):
    cf = json.load(open("ImageProcessingGUI.json", 'r'))

    h,w=image.shape[0:2]

    newimage = np.uint8(np.zeros((h,w,3)))
    

    #graukonversion damit die Konturen mit high und low eingegrenzt werden können
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_binary = cv2.threshold(image_gray,cf["val1"],cf["val2"],cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    bins = 90  #wenn ich es richtig verstanden habe die winkelauflösung. = 360/bins => 360/90=4°
    img_cluster = int(np.ceil(np.sqrt(len(contours))))

    rect_width = w // img_cluster
    rect_height = h // img_cluster

    offsets = [(i, j) for i in range(img_cluster) for j in range(img_cluster)]

    for idx, cont in enumerate(contours[:len(contours)]):
        # Berechne die Momente der Kontur, um den Mittelpunkt zu finden
        M = cv2.moments(cont)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Bestimme die Position im 3x3-Raster
        offset_x, offset_y = offsets[idx]  # Die aktuelle Position im Raster

        # Berechne die obere linke Ecke des Rechtecks
        start_x = offset_x * rect_width
        start_y = offset_y * rect_height

        # Clip die Konturen, um sie innerhalb des jeweiligen Rechtecks zu halten
        # Die einzelnen Rechtecke können eine bestimmte Fläche für die Konturen haben
        cropped_image = newimage[start_y:start_y + rect_height, start_x:start_x + rect_width]

        color = random_colorgen()

        # Zeichne die Konturen und Rechtecke in den entsprechenden Bereich
        for j, pix in enumerate(cont):
            relx = pix[0][0] - cx
            rely = pix[0][1] - cy

            cv2.drawMarker(image_binary,(cx,cy),(255,255,255))

            radius = int(np.sqrt(np.square(relx) + np.square(rely)))
            a = round((np.arctan2(relx, rely) + np.pi) / (2 * np.pi) * bins)

            point1 = (int(a * cropped_image.shape[1] / bins), cropped_image.shape[0] - 1)
            point2 = (int((a + 1) * cropped_image.shape[1] / bins), cropped_image.shape[0] - int(2 * radius))

            # Rechtecke für jedes Shape zeichnen
            cv2.rectangle(cropped_image, point1, point2, color, thickness=cv2.FILLED)


        
        


    return image_binary,newimage


def run(image, result,settings=None):
    data=np.sin(np.arange(0,100,0.5))
    data2,data1 = konturanalyse(image)
    result.append({"name":"Plot","data":data1})
    result.append({"name":"Marker","data":data2})

if __name__ == '__main__':
    matplotlib.use('Agg')
    image=cv2.imread("Images\Objekte.png")
    result=[]
    run(image,result)
    for ele in result:
        if(len(ele["data"].shape)==1):
            fig=plt.figure(num=ele["name"])
            plt.plot(ele["data"])
            fig.tight_layout()
            plt.grid(True)
            plt.xlim(0.0, len(ele["data"])-1)
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer._renderer)
            cv2.imshow(ele["name"],data)
        else:
            cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()