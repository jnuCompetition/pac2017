#coding=utf-8

##################################
#   desc: data.py  
#   author: zhpmatrix
#   date:   2017-07-03

###################################


import pandas as pd
import os

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
	print data.shape
	data.columns=columns 
	data.to_csv(dataName,index=False,encoding='utf-8')

def c2e(labels,x):
	return labels.index(x)

def getTrain(data,pname,label_name,label_value):
	train = data[data.ix[:,'act']==pname]
	train = train.reset_index(drop=True)
	train.loc[:,label_name] = map(lambda x:c2e(label_value,x),train[label_name])
	return train.loc[:,['cmt',label_name]]    	

if __name__ == '__main__':

        # Data generation test
        mergeData()
        
        data = pd.read_csv('data.csv',low_memory=False)
        pname = 'ApplePay'
        label_name = 'ptotal'
        label_value = ['差','中','好']
        train = getTrain(data,pname,label_name,label_value)
        print train.ix[:,'ptotal'].value_counts()
