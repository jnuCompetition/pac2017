#coding=utf-8

import pandas as pd
import numpy as np
from keras.utils import np_utils
from gensim.models import word2vec
import pickle
import jieba
import numpy as np

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

def _word2vec(s,word_dict,word_set,maxlen): 
    	s = [i for i in s if i in word_set]
    	s = s[:maxlen] + ['']*max(0, maxlen-len(s))
    	return list(word_dict[s])

def getStopWords(stopwords_path):
	stopwords = []
	f = open(stopwords_path, 'r')
	while True:
    		line = f.readline()
    		if len(line) == 0:
        		break;
    		line = ''.join(line).strip('\n')
		stopwords.append(line.decode('utf-8'))
	f.close()
	return stopwords

def filterCmt(cutwords,stopwords):
	return [word for word in cutwords if word not in stopwords]

def saveDict(_dict,dict_path):
    _file = open(dict_path,'w')
    pickle.dump(_dict,_file)
    _file.close() 

def show(word_dict):
	print 'Length:',len(word_dict)
	print 'Elements:\n'
	print word_dict

def imbalance(data,label_name):
    subsample_num = 1000
    subsample_min = 5
    df1 = data[data[label_name] == 1].sample(subsample_num)
    
    num0 = data[data[label_name] == 0].shape[0]
    add_0_num = subsample_num - num0
    df0 = data[data[label_name] == 0]
    for i in range(add_0_num/subsample_min):
        _df0 = data[data[label_name] == 0].sample(subsample_min)
        df0 = pd.concat([df0,_df0])
    
    num2 = data[data[label_name] == 2].shape[0]
    add_2_num = subsample_num - num2
    df2 = data[data[label_name] == 2]
    for i in range(add_2_num/subsample_min):
        _df2 = data[data[label_name] == 2].sample(subsample_min)
        df2 = pd.concat([df2,_df2])
    
    data = pd.concat([df0,df1,df2])
    data = data.reset_index(drop=True)
    return data

def splitXY(data,label_name,_min,_max,word_dict_path,word_set_path,stopwords_path):
    '''
        data:
        _min: min value of word freq
        _max: length of word vector
    '''
    data = imbalance(data,label_name)

    # Cut words
    data['words'] = data['cmt'].apply(lambda s: list(jieba.cut(s.replace('\n',''))))
   
    # Word bags
    content = []
    for i in data['words']:
            content.extend(i)	
    
    word_dict = pd.Series(content).value_counts()
    
    # Stop words
    stopwords = getStopWords(stopwords_path)
    indexs = filterCmt(list(word_dict.index),stopwords)
    word_dict = word_dict[indexs]
    
    word_dict = word_dict[word_dict >= _min]
    # Indexing words
    word_dict[:] = range(1, len(word_dict)+1)
    word_dict[''] = 0
    word_set = set(word_dict.index)
    # Dump word set and dict for predicting with jieba
    saveDict(word_dict,word_dict_path)
    saveDict(word_set,word_set_path)
    data['vec'] = data['words'].apply(lambda s: _word2vec(s,word_dict,word_set,_max))
    
    # Shuffle data
    idx = range(len(data))
    np.random.shuffle(idx)
    data = data.loc[idx]
    
    x = np.array(list(data['vec']))
    y = np.array(list(data[label_name]))
    y = y.reshape((-1,1)) 
    y = np_utils.to_categorical(y)
    return x,y,len(word_dict)

def word2vec_(words,model):
    wordMat = []
    for word in words:
        try:
            vec = model[word].tolist()
            wordMat.append(vec)
        except KeyError:
            print word,' Not in vacab!'
    return np.average(wordMat,axis=0)

def splitXY_(data,label_name,stopwords_path):
    data = imbalance(data,label_name)
    # Cut words
    data['words'] = data['cmt'].apply(lambda s: list(jieba.cut(s.replace('\n',''))))
   
    # Stop words
    stopwords = getStopWords(stopwords_path)
    data['words'] = data['words'].apply(lambda s: filterCmt(s,stopwords)) 
    
    # Word2Vec
    model = word2vec.Word2Vec(data['words'].tolist(),min_count=5,size=100,workers=4)
    data['vec']=data['words'].apply(lambda s: word2vec_(s,model))
    
    # Shuffle data
    idx = range(len(data))
    np.random.shuffle(idx)
    data = data.loc[idx]

    x = np.array(list(data['vec']))
    y = np.array(list(data[label_name]))
    y = y.reshape((-1,1)) 
    y = np_utils.to_categorical(y)
    return x,y

if __name__ == '__main__':
       #mergeData()
	
       _max = 100
       _min = 5
       pname = 'ApplePay'
       label_name = 'ptotal'
       label_value = [u'差',u'中',u'好']
    
       word_dict_path = 'wdict'
       word_set_path = 'wset'
       stopwords_path = 'stopwords'
       
       data = pd.read_csv('data.csv',low_memory=False,encoding='utf-8')
       data = getTrain(data,pname,label_name,label_value)
       x,y,dict_len=splitXY(data,label_name,_min,_max,word_dict_path,word_set_path,stopwords_path)
       #x,y=splitXY_(data,label_name,stopwords_path)
       #print x
