#coding=utf-8

##################################
#   desc: data.py  
#   author: zhpmatrix
#   date:   2017-07-04

###################################

FLAG = 1 # Train: 0, Predict: 1

import numpy as np
import pandas as pd
import jieba

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.models import load_model
from keras.utils import np_utils
from data import *
import pickle

import warnings
warnings.filterwarnings("ignore")

def word2vec(s,word_dict,word_set,maxlen): 
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(word_dict[s])

def splitXY(data,label_name,_min,_max,word_dict_path,word_set_path):
    '''
        data:
        _min: min value of word freq
        _max: length of word vector
    '''
    # Cut words
    data['words'] = data['cmt'].apply(lambda s: list(jieba.cut(s)))
    
    # Word bags
    content = []
    for i in data['words']:
            content.extend(i)

    word_dict = pd.Series(content).value_counts()
    word_dict = word_dict[word_dict >= _min]
    
    # Indexing words
    word_dict[:] = range(1, len(word_dict)+1)
    word_dict[''] = 0
    word_set = set(word_dict.index)
    
    # Dump word set and dict for predicting with jieba
    wdict = open(word_dict_path,'w')
    pickle.dump(word_dict,wdict)
    
    wset = open(word_set_path,'w')
    pickle.dump(word_set,wset)
    
    data['vec'] = data['words'].apply(lambda s: word2vec(s,word_dict,word_set,_max))
    
    # Shuffle data
    idx = range(len(data))
    np.random.shuffle(idx)
    data = data.loc[idx]

    x = np.array(list(data['vec']))
    y = np.array(list(data[label_name]))
    y = y.reshape((-1,1)) 
    y = np_utils.to_categorical(y)
    return x,y,len(word_dict)

def train(x,y,params):
    model = Sequential()
    model.add(Embedding(params['input_dim'], 256, input_length=params['input_length']))
    model.add(LSTM(128)) 
    model.add(Dropout(params['dropout']))
    model.add( Dense(3,activation='softmax') )
    # Print model
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    train_num = int( (1-1.0/params['kfolds'])*x.shape[0] )
    model.fit(x[:train_num], y[:train_num], batch_size = params['batch_size'], epochs=params['epochs'])
    model.evaluate(x[train_num:], y[train_num:], batch_size =params['batch_size'])
    model.save(params['save'])

def predict(model,sset,maxlen,wdict,wset):
    '''
        model:
        s: sentence to predict
        maxlen: length of word vector

        return: 0: 差
                1: 中
                2: 优
    '''
    for s in sset:
        s = np.array(word2vec(list(jieba.cut(s)),wdict,wset,maxlen))
        s = s.reshape((1, s.shape[0]))
        print model.predict_classes(s, verbose=0)[0]

if __name__ == '__main__':
    
    _max = 100
    _min = 5
    pname = 'ApplePay'
    label_name = 'ptotal'
    label_value = ['差','中','好']
    model_path = 'model.h5'
    
    word_dict_path = 'wdict'
    word_set_path = 'wset'

    # Test set
    sset = ['产品怎么这么差',
            '买东西很便宜',
            '一定要给别人介绍',
            '这个东西很好呀']

    if FLAG == 0:
        data = pd.read_csv('data.csv')
        # Get data
        data  = getTrain(data,pname,label_name,label_value)
        x,y,dict_len=splitXY(data,label_name,_min,_max,word_dict_path,word_set_path)
        
        # Train model
        params = {}
        params['input_length']=_max
        params['input_dim'] = dict_len
        params['dropout'] = 0.5
        params['batch_size'] = 128
        params['kfolds'] = 5
        params['epochs']=30
        params['save'] = model_path
        
        train(x,y,params)    
    else:
        _wdict = open(word_dict_path,'r')
        wdict = pickle.load(_wdict)

        _wset = open(word_set_path,'r')
        wset = pickle.load(_wset)

        model = load_model(model_path)
        predict(model,sset,_max,wdict,wset)
