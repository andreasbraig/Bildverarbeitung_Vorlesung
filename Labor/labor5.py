import numpy as np
import cv2


def run(image, result,settings=(10,50)):
  svm = cv2.ml.SVM_create()
  svm.setKernel(cv2.ml.SVM_LINEAR)
  #svm.setKernel(cv2.ml.SVM_RBF)
  svm.setType(cv2.ml.SVM_C_SVC)
  svm.setC(settings[0]/10+0.1)
  svm.setGamma(settings[0]/10.0+0.1)
  size=40

  rows=np.zeros((7,3))
  rows[0,:]=[10,10,1]
  rows[1,:]=[0,0,0]
  rows[2,:]=[30,30,0]
  rows[3,:]=[20,10,1]
  rows[4,:]=[20,20,2]
  rows[5,:]=[30,10,2]
  rows[6,:]=[39,0,3]

  input=np.ones((size,size,3))*[0,0,1]
  input[rows[:,0].astype(int),rows[:,1].astype(int),0]=rows[:,2].astype(int)  
  input[rows[:,0].astype(int),rows[:,1].astype(int),1]=rows[:,2].astype(int)  
  input[rows[:,0].astype(int),rows[:,1].astype(int),2]=rows[:,2].astype(int)  

  train = rows[:,0:2].astype(np.float32) 
  response= rows[:,2].astype(int) 

  svm.train(train, cv2.ml.ROW_SAMPLE, response)
  
  mat=np.zeros((size*size,2))
  for i in range(size):
    for j in range(size):
       mat[i*size+j,:]=[i,j]
         
  erg = svm.predict(mat.astype(np.float32))
  erg = erg[1].reshape(size,size)

  cv2.normalize(input,input,0, 1, cv2.NORM_MINMAX)
  input=cv2.resize(input,None,None,10,10,cv2.INTER_NEAREST)
  cv2.normalize(erg,erg,0, 1, cv2.NORM_MINMAX)
  erg=cv2.resize(erg,None,None,10,10,cv2.INTER_NEAREST)

  result.append({"name":"Input","data":input})
  result.append({"name":"Output","data":erg})

if __name__ == '__main__':
    result=[]
    run(None, result)
    for ele in result:
        cv2.imshow(ele["name"],ele["data"])
    cv2.waitKey(-1)
    cv2.destroyAllWindows()