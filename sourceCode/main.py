import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf
import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import Constant
from keras import backend as K
from keras.layers import PReLU

inputData = np.empty((6898,220500))
targetData = np.empty(6898)
root = '../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/'
filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]
i_list = []
rec_annotations = []
rec_annotations_dict = {}

def Extract_Annotation_Data(file_name, root):
    tokens = file_name.split('_')
    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])
    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')
    return (recording_info, recording_annotations)

def slice_data(start, end, raw_data,  sample_rate):
    max_ind = len(raw_data)
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    max_len = 220500
    if (end_ind-start_ind)>max_len:
        return raw_data[start_ind:(start_ind+max_len)]
    elif ((end_ind-start_ind)<max_len):
        return np.concatenate((raw_data[start_ind:end_ind],np.zeros(max_len+start_ind-end_ind)))
    elif (end_ind-start_ind)==max_len:
        return raw_data[start_ind:end_ind]

def getClass(df,index):
    if(df.at[index,'Wheezes']==0 and df.at[index,'Crackles']==0):
        return 0
    elif(df.at[index,'Wheezes']==1 and df.at[index,'Crackles']==0):
        return 1
    elif(df.at[index,'Wheezes']==0 and df.at[index,'Crackles']==1):
        return 2
    elif(df.at[index,'Wheezes']==1 and df.at[index,'Crackles']==1):
        return 3

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
