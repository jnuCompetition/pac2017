#coding=utf-8

import numpy as np
import pandas as pd
import jieba

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.models import load_model

def doc2num(s, maxlen): 
    s = [i for i in s if i in word_set]
    s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    return list(abc[s])


def predict(model,s):
    s = np.array(doc2num(list(jieba.cut(s)), maxlen))
    s = s.reshape((1, s.shape[0]))
    return model.predict_classes(s, verbose=0)[0][0]

if __name__ == '__main__':
    #Train=0  Predict=1
    FLAG = 0
    maxlen = 100
    min_count = 5
    pos = pd.read_excel('../../pos.xls', header=None)
    pos['label'] = 1
    neg = pd.read_excel('../../neg.xls', header=None)
    neg['label'] = 0
    all_ = pos.append(neg, ignore_index=True)
    all_['words'] = all_[0].apply(lambda s: list(jieba.cut(s)))


    content = []
    for i in all_['words']:
            content.extend(i)

    abc = pd.Series(content).value_counts()
    abc = abc[abc >= min_count]
    abc[:] = range(1, len(abc)+1)
    abc[''] = 0
    word_set = set(abc.index)
    if FLAG == 0:


        tmp=doc2num(all_['words'][0],maxlen)

        all_['doc2num'] = all_['words'].apply(lambda s: doc2num(s, maxlen))

        idx = range(len(all_))
        np.random.shuffle(idx)
        all_ = all_.loc[idx]

        x = np.array(list(all_['doc2num']))
        y = np.array(list(all_['label']))
        y = y.reshape((-1,1)) 



        model = Sequential()
        model.add(Embedding(len(abc), 256, input_length=maxlen))
        model.add(LSTM(128)) 
        model.add(Dropout(0.5))
        model.add( Dense(1,activation='sigmoid') )
        
        # Print model
        model.summary()
        
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        batch_size = 128
        train_num = 15000

        model.fit(x[:train_num], y[:train_num], batch_size = batch_size, epochs=30)

        model.evaluate(x[train_num:], y[train_num:], batch_size = batch_size)

        model_name = 'model'+'.h5'
        model.save(model_name)
    
    if FLAG == 1:
        model_name = 'model.h5'
        model = load_model(model_name)
        s = '很好'
        print predict(model,s)
