import librosa
import numpy as np


def np_index(time_string,sr):
    hr=int(time_string[0:2])
    mins=int(time_string[3:5])
    sec=int(time_string[6:8])
    milli_secs=int(time_string[9:])  
    return int((hr*3600+mins*60+sec+milli_secs*0.001)*sr)

f=open('1089.srt','r')
data_array=[]

for line in f:
    data_array.append(line)

#Deleting data_array[0]
data_array=data_array[1:]
time_array=data_array[::4]
sentence_array=data_array[1::4]

start_timestamp=[]
end_timestamp=[]

for i in range(len(time_array)):
    time_array[i]=time_array[i][:-1]
    start_timestamp.append(time_array[i].split(' --> ')[0])
    end_timestamp.append(time_array[i].split(' --> ')[1])

y,sr=librosa.load('1089.wav')


np_sliced_list=[] 
#List of np arrays
for j in range(len(time_array)):
    time_slice=y[np_index(start_timestamp[j],sr):np_index(end_timestamp[j],sr)]
    np_sliced_list.append(time_slice)

mfcc_sliced_list=[]

#List of mfcc np arrays
for j in range(len(np_sliced_list)):
    mfcc_slice=librosa.feature.mfcc(np_sliced_list[j])
    mfcc_sliced_list.append(mfcc_slice)


    
