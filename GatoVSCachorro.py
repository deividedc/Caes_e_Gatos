#import time para usar => #time.sleep(1)
import cv2
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
# Original: %matplotlib inline 
'exec(%matplotlib inline)'

#para cer o diretório
import os
import random
import gc   #Garbage collector for cleaning deleted data from memory
from tqdm import tqdm

#importando a biblioteca Keras
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

#definindo número de epocas
Epocas = 50


#Numero de colunas a serem exibidas as imagens
columns = 5

#leitura dos diretorios de treino e teste
train_dir = 'C:\\Users\\Camila-PC\\Desktop\\reconhecimento_de_imagem\\input\\train'
test_dir = 'C:\\Users\\Camila-PC\\Desktop\\reconhecimento_de_imagem\\input\\test'



#pegando imagens para teste
test_imgs = ['C:\\Users\\Camila-PC\\Desktop\\reconhecimento_de_imagem\\input\\test\\{}'.format(i) for i in os.listdir(test_dir)] 

#pegando imagens para treino
train_imgs = test_imgs = ['C:\\Users\\Camila-PC\\Desktop\\reconhecimento_de_imagem\\input\\train\\{}'.format(i) for i in os.listdir(train_dir)]
# embaralhando cachorros e gatos aleatoriamente
random.shuffle(train_imgs) 


#limpando o lixo de memória para economia
gc.collect()

#Declarando as dimensões das imagens

nrows = 150
ncolumns = 150
channels = 3  #change to 1 if you want to use grayscale image



#Essa função lê e processa as imagens em um formato aceitávelparao modelo
def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = [] # labels
    
    for image in tqdm(list_of_images):
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        #get the labels
        if 'dog' in image:
            y.append(1)
        elif 'cat' in image:
            y.append(0)
        else:
            y.append(0)

    return X, y



#Obtendo dados de treino e rétulo
X, y = read_and_process_image(train_imgs)


import seaborn as sns
del train_imgs
gc.collect()

#Converte a lista em array numpy
X = np.array(X)
y = np.array(y)

#traçar o rótulo para ter certeza de que temos apenas duas classes
sns.countplot(y)
plt.title('Rótulos para gatos e cachorros:')


print("Formato(shape) das imagens de treino:", X.shape)
print("Formato(shape) dos rótulos          :", y.shape)

#dividi os dados em treinamento e conjunto de teste
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

#Limpa memoria
del X
del y
gc.collect()

#obtem o comprimento do trem e os dados de validação
ntrain = len(X_train)
nval = len(X_val)

#Usaremos um tamanho de lote de 32. Observação: o tamanho do lote deve ser um fator de 2. *** 4,8,16,32,64 ... ***
batch_size = 32


#inicialização do modelo

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dropout(0.5))  #Regularização
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  #Função sigmóide no final porque tem apenas duas classes   

#Lets see our model
model.summary()

#Usaremos o otimizador RMSprop com uma taxa de aprendizagem de 0,0001
#Usaremos a perda binary_crossentropy porque é uma classificação binária
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])



#cria a configuração de aumento
#Isso ajuda a evitar overfitting, uma vez que estamos usando um pequeno conjunto de dados
train_datagen = ImageDataGenerator(rescale=1./255,   #Dimensione a imagem entre 0 e 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)

val_datagen = ImageDataGenerator(rescale=1./255)  #Não aumentamos os dados de validação. nós apenas realizamos redimensionamento


#Crie os geradores de imagem
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)




# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> INICIO da Parte de Treinamento
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# #100 passos por época
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=ntrain // batch_size,
#                               epochs=Epocas,
#                               validation_data=val_generator,
#                               validation_steps=nval // batch_size)
# model.save('modelo.h5')

# #vamos traçar o trem e a curva val
# #obter os detalhes do objeto de histórico
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# #Precisão de treinamento e validação
# plt.plot(epochs, acc, 'b', label='Training accurarcy')
# plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
# plt.title('Training and Validation accurarcy')
# plt.legend()
# plt.figure()
# #Perda de treino e validação
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation loss')
# plt.legend()
# plt.show()

# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIM da Parte de Treinamento
# # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>




#%%

from keras.models import load_model
model=load_model('modelo.h5')
test_imgs = ['C:\\Users\\Camila-PC\\Desktop\\reconhecimento_de_imagem\\input\\test{}'.format(i) for i in os.listdir(test_dir)]

TirarFoto=True
ImagensParaAvaliar = 1


if (TirarFoto==False):
    #Now lets predict on the first ImagensParaAvaliar of the test set
    X_test, y_test = read_and_process_image(test_imgs[0:ImagensParaAvaliar]) #Y_test in this case will be empty.
    x = np.array(X_test)
    test_datagen = ImageDataGenerator(rescale=1./255)
    i = 0
    text_labels = []
    plt.figure(figsize=(20,20))
    
    for batch in test_datagen.flow(x, batch_size=1):
        pred = model.predict(batch)
        if pred > 0.5:
            text_labels.append(f'Cachorro {pred}')
        else:
            text_labels.append(f'Gato {pred}')
        #Número de linhas, número de colunas
        plt.subplot((ImagensParaAvaliar / columns) + 1, columns, i + 1)
        plt.title('' + text_labels[i])
        imgplot = plt.imshow(batch[0])
        i += 1
        if i % ImagensParaAvaliar == 0:
            break
    plt.show()
else:   
    camera_port = 0
    file = 'C:\\Users\\Camila-PC\Desktop\\reconhecimento_de_imagem\\input\\test\\aaImagem.bmp'
    while(True):
        #tira foto da WebCam
        camera = cv2.VideoCapture(camera_port)
        retval, img = camera.read()
        cv2.imwrite(file,img)
        camera.release()
        test_imgs = ['C:\\Users\\Camila-PC\Desktop\\reconhecimento_de_imagem\\input\\test\\{}'.format(i) for i in os.listdir(test_dir)]
        X_test, y_test = read_and_process_image(test_imgs[0:ImagensParaAvaliar]) 
        x = np.array(X_test)
        test_datagen = ImageDataGenerator(rescale=1./255)
        i = 0
        text_labels = []
        plt.figure(figsize=(31,31))
        for batch in test_datagen.flow(x, batch_size=1):
            pred = model.predict(batch)
            if pred > 0.7:
                text_labels.append(f'Cachorro {pred}')
            elif pred < 0.3:
                text_labels.append(f'Gato {pred}')
            else:     
                text_labels.append('?')
            plt.subplot((ImagensParaAvaliar / columns) + 1, columns, i + 1)
            plt.title('' + text_labels[i])
            get_ipython().magic('clear')
            imgplot = plt.imshow(batch[0])
            i += 1
            if i % ImagensParaAvaliar == 0:
                break
        plt.show()
# Se necessario apagar a foto tirada pela Web Cam        
#        if(os.path.isfile(file)):
#            os.remove(file)

