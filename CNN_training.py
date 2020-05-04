#This code was made for google collab and TensorFlow 1.x

import sys 
import os 
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.python.keras import optimizers 
#from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.models import Sequential 
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation 
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D 
from tensorflow.python.keras import backend as K 
import io, types

from google.colab import drive
drive.mount('/content/gdrive')
%cd /gdrive

from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

K.clear_session()

os.chdir("/content/gdrive/My Drive/Colab Notebooks/analitica_predictiva")
img_folder= 'data'
data_entrenamiento= 'data/entrenamiento'
data_validacion = 'data/validacion'

from IPython.display import Image
Image ('agenda1660.jpg')

#PARÁMETROS 
epocas=50 
longitud,altura=64,64 
batch_size=40 
pasos=500 
validation_steps=200 
filtrosConv1=32 
filtrosConv2=64 
tamano_filtro1=(3,3) 
tamano_filtro2=(2,2) 
tamano_pool=(2,2) 
clases=3 
lr=0.0005 

#Preparando las imágenes  
entrenamiento_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3, 
    zoom_range=0.3, 
    horizontal_flip=True) 

test_datagen= ImageDataGenerator(rescale= 1./255) 

#Next: generando las imgs que se van a usar para entrenar la red neuronal. 

entrenamiento_generador= entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical') 

validacion_generador= test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura,longitud),
    batch_size=batch_size,
    class_mode='categorical') 

#Next: crear red neuronal convolucional

cnn= Sequential() 
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding = "same",input_shape=(longitud,altura,3),activation='relu')) 
cnn.add(MaxPooling2D(pool_size=tamano_pool)) 

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding = "same", activation='relu'))   
cnn.add(MaxPooling2D(pool_size=tamano_pool)) 
        
cnn.add(Flatten()) 
cnn.add(Dense(256, activation= 'relu')) 
cnn.add(Dropout(0.5)) 
cnn.add(Dense(clases,activation='softmax')) 

#Next: parámetros para optimizar el algoritmo
cnn.compile(loss='categorical_crossentropy', 
           optimizer= optimizers.Adam(lr=lr), 
           metrics=['accuracy']) 


#next: entrenar la red.

cnn.fit_generator(
    entrenamiento_generador, 
    steps_per_epoch=pasos,
    epochs= epocas, 
    validation_data=validacion_generador,
    validation_steps= validation_steps)
        
#next: guardar modelo en un archivo para no tener que entrenarlo cada vez que lo vayamos a usar

target_dir= './modelo/'
if not os.path.exists(target_dir): 
    os.mkdir(target_dir)
cnn.save('./modelo/modelo.h5') 
cnn.save_weights('./modelo/pesos.h5') 

print(entrenamiento_generador.class_indices)
