#--coding=utf-8--
import pandas as pd
import numpy as np
from gensim.models import word2vec
import pickle
import jieba
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  
font = FontProperties(fname=r"fonts/msyh.ttf", size=10)  
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

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
    
    # Balance the data
    train = imbalance(train,label_name)
    
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

def imbalance(data,label_name):
    subsample_num = 1000
    subsample_min = 5
    df1 = data[data[label_name] == 2].sample(subsample_num)
    
    num0 = data[data[label_name] == 1].shape[0]
    add_0_num = subsample_num - num0
    df0 = data[data[label_name] == 1]
    for i in range(int(add_0_num/subsample_min)):
        _df0 = data[data[label_name] == 1].sample(subsample_min)
        df0 = pd.concat([df0,_df0])
    
    num2 = data[data[label_name] == 3].shape[0]
    add_2_num = subsample_num - num2
    df2 = data[data[label_name] == 3]
    for i in range(int(add_2_num/subsample_min)):
        _df2 = data[data[label_name] == 3].sample(subsample_min)
        df2 = pd.concat([df2,_df2])
    
    data = pd.concat([df0,df1,df2])
    data = data.reset_index(drop=True)
    return data


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

def noiseLabel(sample):
    if sample == "其他":
        return 2
    else:
        return 1

def getTrain_(data,label_name):
    data.loc[:,"noise"] = list(map(lambda x: noiseLabel(x),data["act"])) 
    # Balance the data
    train = _imbalance(data,label_name)
    
    # Test
    trainNum = 4000
    train = train.sample(trainNum)
    
    res=train.loc[:,['cmt',label_name]]
    _res = np.array(res)
    _data = []
    for elem in _res:
        _data.append((elem[0],elem[1]))
    return _data

def _imbalance(data,label_name):
    """
        desc: 'act' == '其他'(label = 1),else (label = 0)
    """
    subsample_num = 19000
    subsample_min = 100
    df1 = data[data[label_name] == 2].sample(subsample_num)
    
    num0 = data[data[label_name] == 1].shape[0]
    add_0_num = subsample_num - num0
    df0 = data[data[label_name] == 1]
    for i in range(int(add_0_num/subsample_min)):
        _df0 = data[data[label_name] == 1].sample(subsample_min)
        df0 = pd.concat([df0,_df0])
    
    data = pd.concat([df0,df1])
    data = data.reset_index(drop=True)
    return data

def ruleCmp(data):
    posVals = {}
    acts = ["银联62","ApplePay","银联钱包","云闪付"]
    attrs = ["fav","eval","pser","adv","time","ptotal"]
    attrName = ["优惠力度","应用评价","服务评价","活动宣传","活动时间","整体评价"]
    for i in range(len(acts)):
        vals = []
        for j in range(len(attrs)):
            act = data[data["act"] == acts[i]]
            totalNum = act.shape[0]
            _act = act[attrs[j]].value_counts()
            if attrs[j] == "adv" or attrs[j] == "time":
                val = float(_act["充分"]) / totalNum
            else:
                val = float( _act["好"] )/totalNum
            vals.append(val)
        posVals[acts[i]]=vals
    
    posDf=pd.DataFrame(posVals,index=attrs)
    plt.figure(figsize=(8,8))
    posDf.plot(kind="bar")
    plt.title("相同属性不同银联产品对比",fontproperties=font)
    plt.xlabel("银联产品属性",fontproperties=font)
    plt.ylabel("积极百分比值",fontproperties=font)
    plt.savefig("pos.png")
    
    for act in acts: 
        plt.figure(figsize=(7,7))
        plt.title(act+"不同属性积极百分比",fontproperties=font)
        plt.xlabel("产品属性",fontproperties=font)
        plt.ylabel("积极百分比值",fontproperties=font)
        posDf[act].plot(kind="bar")
        plt.savefig(act+".png")
    return posDf

def to_num(label):
    if label == "好" or label == "充分":
        return 0
    elif label == "中" or label == "中等":
        return 1
    else:
        return 2
def corr(data):
    acts = ["银联62","ApplePay","银联钱包","云闪付"]
    attrs = ["fav","eval","pser","adv","time","ptotal"]
    for attr in attrs:
        data.loc[:,attr] = list(map(lambda x:to_num(x),data[attr]))
    corrVals = {}
    for act in acts:
        corrs = []
        _data = data[data["act"] == act]
        for i in range(len(attrs)-1):
            corrs.append(_data[attrs[i]].corr(_data["ptotal"]))
        corrVals[act] = corrs
    
    corrDf=pd.DataFrame(corrVals,index=["fav","eval","pser","adv","time"])
    plt.figure(figsize=(8,8))
    corrDf.plot(kind="bar")
    plt.title("相同属性不同银联产品相关系数对比",fontproperties=font)
    plt.xlabel("银联产品属性",fontproperties=font)
    plt.ylabel("相关系数",fontproperties=font)
    plt.savefig("corr.png")
    
    for act in acts: 
        plt.figure(figsize=(7,7))
        plt.title(act+"不同属性相关系数",fontproperties=font)
        plt.xlabel("产品属性",fontproperties=font)
        plt.ylabel("相关系数",fontproperties=font)
        corrDf[act].plot(kind="bar")
        plt.savefig(act+".png")
    return corrVals

if __name__ == '__main__':
    
    #data = getStopWords("stopwords")
    #cutWords = "张海鹏把这个处理一下呗！"
    #_cutWords = list(jieba.cut(cutWords.replace('\n','')))
    #_data = filterCmt(_cutWords,data)
    #print (_data)
    
    raw_data = pd.read_csv('data.csv',low_memory=False,encoding='utf-8')
    #posDf = ruleCmp(raw_data)
    data = corr(raw_data)    
    #data = getTrain(raw_data,'ApplePay','ptotal',[u'差',u'中',u'好'])
    #w2v=get_w2v(data)
