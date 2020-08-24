'''
    Created on Jul 26, 2020
    @author: frank
'''

import csv
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,svm,tree
import pandas as pd
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Flatten,Embedding,LSTM


DATA = []

workload_eum = {}
framework_eum = {}
datasize_eum = {}
vmtype_eum = {}



stdlen = 80
BATCH_SIZE = 64
NB_EPOCH = 3
VALIDATION_SPLIT = 0.2

def show_files(path):
    file_list = os.listdir(path)
    for file in file_list:
        cur_path = os.path.join(path, file)
        if os.path.isdir(cur_path):
            show_files(cur_path)
        else:
            datafile.append(cur_path)


def show_files_1(path, all_files):
    for root,dirs,files in os.walk(path):
        for file in files:
            all_files.append(os.path.join(root,file))
    return all_files

def main():
    path = "/Users/huyi/Desktop/scout/dataset/osr_single_node"
    json_files = show_files_1(path, [])
    #show_files(path)
    #json_files = datafile
    workload = []
    framework = []
    vm_type = []
    x_data = []
    y_data = []
    num = 0
    max_time = 770
    for json_file in json_files:
        #         print json_file
        base_name = os.path.basename(json_file)
        dir_name = os.path.dirname(json_file)
        base_dir_name = os.path.basename(dir_name)
        if base_name == 'report.json':
            with open(json_file, 'r') as load_f:
                contents = json.load(load_f)
                if contents.get('completed') == False:
                    continue
                else:
                    y_data.append('%.2f' % float(contents.get('elapsed_time')))
                    '''
                    one_data.append('%.2f' % float(contents.get('elapsed_time')))
                    
                    wl = contents.get('workload')
                    if wl not in workload_eum.keys():
                        workload_eum[wl] = len(workload_eum)
                    one_data.append( workload_eum[wl])
                    
                    fw = contents.get('framework')
                    if fw not in framework_eum.keys():
                        framework_eum[fw] = len(framework_eum)
                    one_data.append(framework_eum[fw])
                    
                    ds = contents.get('datasize')
                    if ds not in datasize_eum.keys():
                        datasize_eum[ds] = len(datasize_eum)
                    one_data.append(datasize_eum[ds])
                    
                    vt = base_dir_name.split('_')[0]
                    if vt not in vmtype_eum.keys():
                        vmtype_eum[vt] = len(vmtype_eum)
                    one_data.append(vmtype_eum[vt])
                    
                    vm_type.append(base_dir_name.split('_')[0])
                    workload.append(contents.get('workload'))
                    framework.append(contents.get('framework'))
                    '''
            if os.path.exists(os.path.join(dir_name, 'sar.csv')):
                with open(os.path.join(dir_name, 'sar.csv'), 'r') as load_f:
                    reader = csv.DictReader(load_f)
                    count = 0
                    x_data.append([])
                    for i in reader:
                        x_data[num].append([])
                        for key in i.keys():
                            if key != 'timestamp':
                                x_data[num][count].append(float(i[key]))
                        count += 1
                    keylen = len(x_data[num][count-1])
                    while count < max_time:
                        x_data[num].append([])
                        for j in range(keylen):
                            x_data[num][count].append(0)
                        count += 1
                    num += 1


    x_data = np.array(x_data,dtype=np.float32)
    y_data = np.array(y_data,dtype=np.float32)
    
    for i in range(len(x_data)):
        xmeans = np.mean(x_data[i],0)
        xvar = np.var(x_data[i],0)
        for j in range(len(xvar)):
            if(xvar[j] == 0):
                xvar[j] = 1
        x_data[i] = (x_data[i] - xmeans) / xvar
    ymean = np.mean(y_data)
    y_data = y_data - ymean


    validation_rate = 0.8
    x_train = x_data[0:int(validation_rate*num)]
    y_train = y_data[0:int(validation_rate*num)]
    x_test = x_data[int(validation_rate*num):]
    y_test = y_data[int(validation_rate*num):]

    x_train = np.array(x_train,dtype=np.float32)
    x_test = np.array(x_test,dtype=np.float32)
    y_train = np.array(y_train,dtype=np.float32)
    y_test = np.array(y_test,dtype=np.float32)



    model = Sequential()
    model.add(LSTM(12,return_sequences=True))
    model.add(Dropout(0.6))
    model.add(Flatten(name = "flatten_0"))
    model.add(Dense(1))

    optimizer = Adam()
    model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['accuracy'])

              
    history = model.fit(x_train,y_train,batch_size=BATCH_SIZE, epochs=NB_EPOCH,validation_split=VALIDATION_SPLIT)

    model.summary()
    
    layer_model = Model(inputs=model.input,
                    outputs=model.get_layer("flatten_0").output)

    output = layer_model.predict(x_data)

    with open("lstm.txt","w") as fo:
        for i in range(len(output)):
            fo.write(output[i])
            fo.write("\n")

if __name__ == '__main__':
    main()
