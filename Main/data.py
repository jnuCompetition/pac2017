#coding=utf-8

##################################
#   desc: data.py  
#   author: zhpmatrix
#   date:   2017-07-03

###################################


import pandas as pd
import os
import numpy as np
from keras.utils import np_utils
import pickle
import jieba

def mergeData():
	dataName = 'data.csv'
        dataDir = 'data/'
	
        files_1= ['第一批训练数据-1（13424） .xls','第一批训练数据-2（9824） .xls',
		  '第一批训练数据-3（2338）.xls','第一批训练数据-4（8696）.xls',
		  '第一批训练数据-5（12965）.xls','第一批训练数据-6（22043）.xls']
	files_2 = ['第二批训练数据-10（266）.xls','第二批训练数据-11（403）.xls',
		  '第二批训练数据-12（510）.xls','第二批训练数据-13（1158）.xls',
		  '第二批训练数据-14（1178）.xls','第二批训练数据-15（4032）.xls',
		  '第二批训练数据-16（8981）.xls','第二批训练数据-1（11）.xls',
		  '第二批训练数据-3（36）.xls','第二批训练数据-4（49）.xls',
		  '第二批训练数据-5（67）.xls','第二批训练数据-6（98）.xls',
		  '第二批训练数据-7（179）.xls','第二批训练数据-8（221）.xls','第二批训练数据-9（253）.xls']
        columns=['cmt','act','fav','eval','penv','pser','adv','time','ptotal','senv','qua','sser','stotal','link']
        data=pd.DataFrame()
	files = [files_1,files_2]
	for i in range(len(files)):
		for fname in files[i]:
	    		# Read data
	    		tdata = pd.read_excel(dataDir+fname,header=None)
	    		# Remove header lines
	    		tdata = tdata[2:]
			# Add link column for first batch data
			if i == 0:
	    			tdata.loc[:,13]=None
			data = data.append(tdata,ignore_index=True)
	# Reset columns as English name
	data.columns=columns 
	data.to_csv(dataName,index=False,encoding='utf-8')

def c2e(labels,x):
	return labels.index(x)

def getTrain(data,pname,label_name,label_value):
	train = data[data.ix[:,'act']==pname]
	train = train.reset_index(drop=True)
	train.loc[:,label_name] = map(lambda x:c2e(label_value,x),train[label_name])
	return train.loc[:,['cmt',label_name]]    	

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

if __name__ == '__main__':

	_max = 100
   	_min = 5
    	pname = 'ApplePay'
    	label_name = 'ptotal'
    	label_value = ['差','中','好']
    
    	word_dict_path = 'wdict'
    	word_set_path = 'wset'
        data = pd.read_csv('data.csv')
        data  = getTrain(data,pname,label_name,label_value)
	x,y,dict_len=splitXY(data,label_name,_min,_max,word_dict_path,word_set_path)
        print x
