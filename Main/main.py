#coding=utf-8

FLAG = 0 # Train: 0, Predict: 1

import numpy as np
import pandas as pd
import jieba
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.models import load_model
from keras.utils import np_utils
from sklearn import metrics
#import data
from data import *
import pickle
import warnings
warnings.filterwarnings("ignore")

def find_all_index(arr,item):
    return [i for i,a in enumerate(arr) if a == item]
def get_all_value(arr,index):
    return [arr[idx] for idx in index]
def show_metrics(real,pred):
    for label in range(0,3):
        index = find_all_index(real,item=label)
        _real = get_all_value(real,index)
        _pred = get_all_value(pred,index)
        precision = metrics.precision_score(_real,_pred,average='micro')
        #recall = metrics.recall_score(_real,_pred,average='micro')
        #f1_score = metrics.f1_score(_real,_pred,average='micro')
        print 'Test Results {}:'.format(label)
        #print 'Real:',_real
        #print 'Pred:',_pred
        print 'Precision: {}'.format(precision)
        #print 'Recall: {}'.format(recall)
        #print 'F1_score: {}'.format(f1_score)

def vec2num(categorical_label):
    labels = []
    for label in categorical_label.tolist():
        labels.append(label.index(1))
    return labels

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
    #results = []
    #_results = {}
    #results = model.evaluate(x[train_num:], y[train_num:], batch_size =params['batch_size'],verbose=True)
    #_results['loss'] = results[0]
    #_results['acc'] = results[1]
    #print _results
    
    pred = model.predict_classes(x[train_num:], verbose=False)
    pred = pred.tolist()
    real = vec2num(y[train_num:])
     
    show_metrics(real,pred)
    
    model.save(params['save'])

def _train(x,y,params):
    model = Sequential()
    #model.add(Embedding(params['input_dim'], 256, input_length=params['input_length']))
    #model.add(LSTM(128)) 
    
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    model.add(LSTM(128,input_shape=(x.shape[1],x.shape[2])))
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
    #results = []
    #_results = {}
    #results = model.evaluate(x[train_num:], y[train_num:], batch_size =params['batch_size'],verbose=True)
    #_results['loss'] = results[0]
    #_results['acc'] = results[1]
    #print _results
    
    pred = model.predict_classes(x[train_num:], verbose=False)
    pred = pred.tolist()
    real = vec2num(y[train_num:])
    show_metrics(real,pred)
    model.save(params['save'])

def showCutWords(cutwords):
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
    stopwords = data.getStopWords(stopwords_path)
    for s in sset:
        # Remove stop words
        cutwords = data.filterCmt( list(jieba.cut(s.replace('\n',''))),stopwords )
        #showCutWords(cutwords)
        s = np.array( data._word2vec(cutwords,wdict,wset,maxlen) )
        s = s.reshape((1, s.shape[0]))
        res = model.predict_classes(s, verbose=False)[0]
        if res == 0:
            print '差'
        elif res == 1:
            print '中'
        else:
            print '好'

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
            'applepay只能是银联吗？ wlmouse 发表于 2016-2-26 09:57 地区变成美国，能绑外卡不。中国即可绑就像之前地区改美国也能绑银联卡一样',
            '关于目前的部分Pos机对于ApplePay闪付的一个缺陷 505597029 发表于 2016-2-22 14:54 体验写的很仔细，点赞…不过很多收银员都不会用闪付才是真的缺陷…',
            'Samsung Pay在中国注定干不过Apple Pay的原因 我看三星根本没想这么多。怎么样它都占据了手机器件，还有半导体芯片的上游，这个行业在他就会一直爽下去。他只要保证时刻不掉队即可。过几年几十年又出另一个苹果或者另几个，总有哪个死了，不过三星怎么都不会死。因为大家都得用它的器件。']

    if FLAG == 0:
        data = pd.read_csv('data.csv')
        # Get data
        data  = getTrain(data,pname,label_name,label_value)
        #x,y,dict_len = splitXY(data,label_name,_min,_max,word_dict_path,word_set_path,stopwords_path)
        x,y = splitXY_(data,label_name,stopwords_path)
        
        # Train model
        params = {}
        params['input_length']=_max
        params['dropout'] = 0.2
        params['batch_size'] = 128
        params['kfolds'] = 5
        params['epochs']= 80
        params['save'] = model_path
        
        #params['input_dim'] = dict_len
        _train(x,y,params)    
    
    else:
        _wdict = open(word_dict_path,'r')
        wdict = pickle.load(_wdict)

        _wset = open(word_set_path,'r')
        wset = pickle.load(_wset)

        model = load_model(model_path)
        predict(model,sset,_max,wdict,wset,stopwords_path)
