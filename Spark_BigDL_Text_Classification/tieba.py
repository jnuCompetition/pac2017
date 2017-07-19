#coding=utf-8

import pyhdfs  
import time
import urllib
import re
from bs4 import BeautifulSoup

class Tool:
    #去除img标签,7位长空格
    removeImg = re.compile(b'<img.*?>| {7}|')
    #删除超链接标签
    removeAddr = re.compile(b'<a.*?>|</a>')
    #把换行的标签换为\n
    replaceLine = re.compile(b'<tr>|<div>|</div>|</p>')
     #将表格制表<td>替换为\t
    replaceTD= re.compile(b'<td>')
    #把段落开头换为\n加空两格
    replacePara = re.compile(b'<p.*?>')
    #将换行符或双换行符替换为\n
    replaceBR = re.compile(b'<br><br>|<br>')
    #将其余标签剔除
    removeExtraTag = re.compile(b'<.*?>')
    def replace(self,x):
        x = re.sub(self.removeImg,b"",x)
        x = re.sub(self.removeAddr,b"",x)
        x = re.sub(self.replaceLine,b"\n",x)
        x = re.sub(self.replaceTD,b"\t",x)
        x = re.sub(self.replacePara,b"\n    ",x)
        x = re.sub(self.replaceBR,b"\n",x)                                 
        x = re.sub(self.removeExtraTag,b"",x)
        return x.strip()

class BDTB:
    def __init__(self,baseUrl,seeLZ,floorTag,t):
        self.baseURL = baseUrl
        self.seeLZ = '?see_lz='+str(seeLZ)
        self.tool = Tool() 
        self.file = None
        self.t = t
        self.floor = 1
        self.floorTag = floorTag
        self.time = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        #self.hdfsdir = "/user/kawayi/cmts/"+str(t)
        self.local_dir = "cmts/"+self.time+".txt"
    def getUrls(self):
        req = urllib.request.Request(self.baseURL)
        content = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(content,"html.parser")
        alinks = soup.find_all('a')
        urls=[]
        for alink in alinks:
            href = str(alink.get('href'))
            link = re.compile("/p/\d{10}") 
            if link.findall(href):
                url = link.findall(href)
                urls+=url
        return urls
    def getPage(self,pageNum):
        try:
            url = self.baseURL+self.seeLZ+'&pn='+str(pageNum)
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            return response.read()
        except urllib.error.URLError as e:
            print(e.reason)
    def getPageNum(self,page):
        pattern = re.compile(b'<li class="l_reply_num.*?</span>.*?<span.*?>(.*?)</span>',re.S)
        result = re.search(pattern,page)
        if result:
            return result.group(1).strip()
        else:
            return None

    def getContent(self,page):
        pattern = re.compile(b'<div id="post_content_.*?>(.*?)</div>',re.S)
        items = re.findall(pattern,page)
        contents = []
        for item in items:
            content = b"\n"+self.tool.replace(item)+b"\n"
            contents.append(content)
        return contents

    def writeData(self,contents):
        filename=self.local_dir
        self.file=open(filename,"w+")
        for item in contents:
            self.file.write(str(item,"utf-8"))
            self.floor +=1
    
    def start(self):
        indexPage=self.getPage(1)
        #fs = pyhdfs.HdfsClient("localhost:50070")
        pageNum=self.getPageNum(indexPage)
        if pageNum == None:
            print("URL not found!")
        try:
            print("Total pages=",str(pageNum))
            for i in range(1,int(pageNum)+1):
                page = self.getPage(i)
                contents = self.getContent(page)
                self.writeData(contents)
            #fs.copy_from_local(self.local_dir,self.hdfsdir)
        except IOError:
            print("Fail to write!")

if __name__ == "__main__":
    
    baseURL='https://tieba.baidu.com/p/5139811130'
    floorTag=1
    seeLZ=0
    bdtb=BDTB(baseURL,seeLZ,floorTag,t)
    bdtb.start()
