#coding=utf-8

##################################
#   desc: data.py  
#   author: zhpmatrix
#   date:   2017-07-05

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
    model.fit(x[:train_num], y[:train_num], batch_size = params['batch_size'], epochs=params['epochs'],verbose=True)
   
    # Get loss and acc from test set
    results = []
    _results = {}
    results = model.evaluate(x[train_num:], y[train_num:], batch_size =params['batch_size'],verbose=True)
    _results['loss'] = results[0]
    _results['acc'] = results[1]
    print _results

    model.save(params['save'])

def showCutWords(cutwords):
     print '----------------'
     for word in cutwords:
         print word.encode('utf-8')

def predict(model,sset,maxlen,wdict,wset,stopwords_path):
    '''
        model:
        s: sentence to predict
        maxlen: length of word vector

        return: 0: 差
                1: 中
                2: 优
    '''
    stopwords = getStopWords(stopwords_path)
    for s in sset:
        # Remove stop words
        cutwords = filterCmt( list(jieba.cut(s.replace('\n',''))),stopwords )
        showCutWords(cutwords)
        s = np.array(word2vec(cutwords,wdict,wset,maxlen))
        s = s.reshape((1, s.shape[0]))
        print model.predict_classes(s, verbose=False)[0]

if __name__ == '__main__':
    
    _max = 100
    _min = 5
    pname = 'ApplePay'
    label_name = 'ptotal'
    label_value = ['差','中','好']
    
    word_dict_path = 'wdict'
    word_set_path = 'wset'
    model_path = 'model.h5'
    stopwords_path = 'stopwords'

    # Test set
    sset = ['一个严肃的问题，招行双币一卡通，VISA+银联，是否可以用卡号+有效期通过visa消费 借记卡也可以这么消费啊？太不安全了',
            '买东西很便宜',
            '一定要给别人介绍',
            'ApplePay这个东西很好呀']

    if FLAG == 0:
        data = pd.read_csv('data.csv')
        # Get data
        data  = getTrain(data,pname,label_name,label_value)
        x,y,dict_len=splitXY(data,label_name,_min,_max,word_dict_path,word_set_path,stopwords_path)
        
        # Train model
        params = {}
        params['input_length']=_max
        params['input_dim'] = dict_len
        params['dropout'] = 0.5
        params['batch_size'] = 128
        params['kfolds'] = 5
        params['epochs']=2
        params['save'] = model_path
        
        train(x,y,params)    
    
    else:
        _wdict = open(word_dict_path,'r')
        wdict = pickle.load(_wdict)

        _wset = open(word_set_path,'r')
        wset = pickle.load(_wset)

        model = load_model(model_path)
        predict(model,sset,_max,wdict,wset,stopwords_path)
