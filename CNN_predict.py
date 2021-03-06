#código utilizado para la predicción del modelo que se obtuvo usando CNN_train. 

import numpy as np 
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model 
from tensorflow.python.keras.models import load_model

longitud,altura =64,64 
modelo= './modelo/modelo.h5' 
pesos_modelo='./modelo/pesos.h5'
cnn = load_model(modelo) 
cnn.load_weights(pesos_modelo)


def predict(file): 
    x= load_img(file,target_size=(longitud,altura))
    x= img_to_array(x) 
    x= np.expand_dims(x,axis=0)
    array = cnn.predict(x)
    result= array[0] 
    answer= np.argmax(result) 
    if answer == 0:
        print("pred: Cinta")
    elif answer == 1:
        print("pred: agenda")
    elif answer == 2:
        print("pred: armario")
        
    return answer
