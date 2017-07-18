#coding=utf-8

import pandas as pd
import numpy as np
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
            tdata = pd.read_excel(dataDir+fname,header=None)
            tdata = tdata[2:]
            if i == 0:
                tdata.loc[:,13]=None
            data = data.append(tdata,ignore_index=True)
    data.columns=columns
    data.to_csv(dataName,index=False,encoding='utf-8')

def c2e(labels,x):
	return labels.index(x)+1

def getTrain(data,pname,label_name,label_value):
    train = data[data.ix[:,'act']==pname]
    train = train.reset_index(drop=True)
    train.loc[:,label_name] = list(map(lambda x:c2e(label_value,x),train[label_name]))
    res=train.loc[:,['cmt',label_name]]
    _res = np.array(res)
    _data = []
    for elem in _res:
        _data.append((elem[0],elem[1]))
    return _data

def getStopWords(stopwords_path):
    stopwords = []
    f = open(stopwords_path,"r")
    while True:
        line = f.readline()
        if len(line) == 0:
            break
        line = "".join(line).strip("\n")
        stopwords.append(line)
    f.close()
    return stopwords

def filterCmt(cutwords,stopwords):
	return [word for word in cutwords if word not in stopwords]

#def imbalance(data,label_name):
#    subsample_num = 1000
#    subsample_min = 5
#    df1 = data[data[label_name] == 1].sample(subsample_num)
#    
#    num0 = data[data[label_name] == 0].shape[0]
#    add_0_num = subsample_num - num0
#    df0 = data[data[label_name] == 0]
#    for i in range(add_0_num/subsample_min):
#        _df0 = data[data[label_name] == 0].sample(subsample_min)
#        df0 = pd.concat([df0,_df0])
#    
#    num2 = data[data[label_name] == 2].shape[0]
#    add_2_num = subsample_num - num2
#    df2 = data[data[label_name] == 2]
#    for i in range(add_2_num/subsample_min):
#        _df2 = data[data[label_name] == 2].sample(subsample_min)
#        df2 = pd.concat([df2,_df2])
#    
#    data = pd.concat([df0,df1,df2])
#    data = data.reset_index(drop=True)
#    return data
#

def get_w2v(data):
    w2v = {}
    all_cmts = []
    for v in data:
        all_cmts.append(list(jieba.cut(v[0].replace('\n',''))))
    vec_len = 50
    model = word2vec.Word2Vec(all_cmts,min_count=5,size=vec_len,workers=4)
    for key in model.wv.vocab:
        w2v[key]=model[key].tolist()
    w2v['##']=[0]*vec_len
    return w2v


if __name__ == '__main__':
    
    data = getStopWords("stopwords")
    cutWords = "张海鹏把这个处理一下呗！"
    _cutWords = list(jieba.cut(cutWords.replace('\n','')))
    _data = filterCmt(_cutWords,data)
    print (_data)
    
    #mergeData()
    
    #raw_data = pd.read_csv('data.csv',low_memory=False,encoding='utf-8')
    #data = getTrain(raw_data,'ApplePay','ptotal',[u'差',u'中',u'好'])
    #w2v=get_w2v(data)
