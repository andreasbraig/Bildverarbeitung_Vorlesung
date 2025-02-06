import numpy as np
import cv2


def run(image, result,settings=(10,50)):
  svm = cv2.ml.SVM_create()
  #svm.setKernel(cv2.ml.SVM_LINEAR)
  svm.setKernel(cv2.ml.SVM_RBF)
  svm.setType(cv2.ml.SVM_C_SVC)
  svm.setC(settings[0]/10+0.1)
  svm.setGamma(settings[0]/10.0+0.1)

  h,w=image.shape[0:2]

  #Definition der Farben und klassen immer oberer und unterer Wert 

  rows=np.zeros((12,4))
  #Grün als Klasse 0
  rows[0,:]=[65,78,4,0]
  rows[1,:]=[43,62,0,0]

  #Gelb als Klasse 1
  rows[2,:]=[63, 135, 142,1]
  rows[3,:]=[35, 104, 112,1]

  #Orange als Klasse 2
  rows[4,:]=[40,  63, 141,2]
  rows[5,:]=[13,  40, 127,2]

  #Rot als Klasse 3 
  rows[6,:]=[23,23,89,3]
  rows[7,:]=[3,6,61,3]

  #Blau als Klase 4
  rows[8,:]=[104,44,8,4]
  rows[9,:]=[66,25,2,4]

  #Hintergrund
  rows[10,:]=[163, 149, 130,5]
  rows[11,:]=[139, 125, 107,5]

  #Training -> Dimensionen jetzt auf 3 erhöht 
  train = rows[:,0:3].astype(np.float32) 
  response= rows[:,3].astype(int) 

  svm.train(train, cv2.ml.ROW_SAMPLE, response)
  
  print(image.shape)
  mat=image.reshape(-1,3)
  print(mat.shape)
         
  #Gebe dem ganzen jeden Pixel und lasse ihn Predicten was er ist 
  erg = svm.predict(mat.astype(np.float32))
  erg = erg[1].reshape(h,w)
  #erweiterung um Color Result 
  color_result = np.uint8(train[np.int32(2*erg),:])
  print(color_result.shape)

  cv2.normalize(erg,erg,0, 1, cv2.NORM_MINMAX)
  #erg=cv2.resize(erg,None,None,10,10,cv2.INTER_NEAREST)

  result.append({"name":"color_result","data":color_result})
  result.append({"name":"Output","data":erg})

if __name__ == '__main__':
    result=[]
    run(None, result)
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(-1)
    cv2.destroyAllWindows()