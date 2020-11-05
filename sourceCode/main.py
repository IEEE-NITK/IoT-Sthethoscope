import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
import soundfile as sf
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.initializers import Constant
from keras import backend as K
from keras import regularizers
from keras.layers import PReLU
import cv2
import librosa

inputData = np.empty((6898,50000))
targetData = np.empty(6898)

root = '../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
i_list = []
rec_annotations = []
rec_annotations_dict = {}
imageInputData = np.empty((6898,13,98))

def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)

def slice_data(start, end, raw_data,  sample_rate):
    max_ind = len(raw_data)
    new_sample_rate = 10000
    new_raw_data = signal.resample(raw_data,int(max_ind*new_sample_rate/sample_rate))
    new_max_ind = len(new_raw_data)
    start_ind = min(int(start * new_sample_rate), new_max_ind)
    end_ind = min(int(end * new_sample_rate), new_max_ind)
    max_len = 50000
    if (end_ind-start_ind)>max_len:
        #print('1')
        return new_raw_data[start_ind:(start_ind+max_len)]
    elif ((end_ind-start_ind)<max_len):
        #print('2')
        return np.concatenate((new_raw_data[start_ind:end_ind],np.zeros(max_len+start_ind-end_ind)))
    elif (end_ind-start_ind)==max_len:
        #print('3')
        return new_raw_data[start_ind:end_ind]
    
def getClass(df,index):
    if(df.at[index,'Wheezes']==0 and df.at[index,'Crackles']==0):
        return 0
    elif(df.at[index,'Wheezes']==1 and df.at[index,'Crackles']==0):
        return 1
    elif(df.at[index,'Wheezes']==0 and df.at[index,'Crackles']==1):
        return 2
    elif(df.at[index,'Wheezes']==1 and df.at[index,'Crackles']==1):
        return 3
# Model configuration
img_width, img_height = 20, 98
batch_size = 50
no_epochs = 100
no_classes = 4
validation_split = 0.25
verbosity = 1
input_shape = (img_width, img_height,1)

for s in filenames:
    (i,a) = Extract_Annotation_Data(s, root)
    i_list.append(i)
    rec_annotations.append(a)
    rec_annotations_dict[s] = a
recording_info = pd.concat(i_list, axis = 0)
recording_info.head()

l=0
for i in rec_annotations_dict:
    j = rec_annotations_dict[i]
    for k in range(j.shape[0]):
        data,sampleRate = sf.read(root+i+'.wav')
        inputData[l] = slice_data(j.at[k,'Start'],j.at[k,'End'], data, sampleRate)
        targetData[l] = getClass(j,k)
        l=l+1

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu',kernel_regularizer=regularizers.l1(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu',kernel_regularizer=regularizers.l1_l2(0.01)))
model.add(Dense(no_classes, activation='softmax',))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# Fit data to model
history = model.fit(inputImageData, targetData, batch_size=batch_size, epochs=40, verbose=verbosity, validation_split=validation_split)
# Visualize model history
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('training / validation accuracies')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('training / validation loss values')
plt.ylabel('Loss value')
plt.xlabel('Epoch')
plt.legend(loc="upper left")
plt.show()
