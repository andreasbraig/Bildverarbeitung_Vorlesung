import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

def local_max(data):
    count = 0 
    for r in range(len(data)):
        if data[(r-1)%len(data)] <= data[r] and data[r] > data[(r+1) % len(data)]:
            count += 1

    return count

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

        cv2.drawMarker(image_binary,(cx,cy),(255,255,255)) # der Marker muss ja nur einmal pro Kontur gezeichnet werden und nicht für jeden Pixel wieder neu :)


        radius = np.zeros((len(cont), 1))
        a = np.zeros((len(cont), 1))
        step = 3 #len(cont) / bins
        # Zeichne die Konturen und Rechtecke in den entsprechenden Bereich
        for j, pix in enumerate(cont):
            relx = pix[0][0] - cx
            rely = pix[0][1] - cy

            radius[j] = int(np.sqrt(np.square(relx) + np.square(rely)))
            a[j] = round((np.arctan2(relx, rely) + np.pi) / (2 * np.pi) * bins)

        for i in range(5):
            radius=(np.roll(radius,1)+radius+np.roll(radius,-1))/3

        for r in range(len(radius)):
            rad = radius[r]
            point1 = (int(a[r] * cropped_image.shape[1] / bins), cropped_image.shape[0] - 1)
            point2 = (int((a[r] + 1) * cropped_image.shape[1] / bins), cropped_image.shape[0] - int(2 * rad))

            # Rechtecke für jedes Shape zeichnen
            cv2.rectangle(cropped_image, point1, point2, color, thickness=cv2.FILLED)
        
        cv2.putText(img= cropped_image,text= str(local_max(radius)), color = (255,255,255), org= (10,10), fontFace=1, fontScale=1)

        print(radius)
        # radius_cat = radius + radius + radius 
        # corner_count = 0


        # for x, r in enumerate(radius_cat):

        #     if len(radius_cat) > x + 2 & radius_cat[x+1] > r & radius_cat[x+1] > radius_cat[x+2]:
        #         corner_count += 1


        # corner_count = int(corner_count / 3)
        # print(corner_count)


        
        


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