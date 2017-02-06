#encoding:utf-8
import urllib2
import re
from bs4 import BeautifulSoup

#建立列表储存url和对应的标签名
url = []
name= []
#建立集合， 将爬取过的页面放入，避免才重复
urlset =set("")
#用于筛选的正则表达式
source = ".银行"
temp = source.decode('utf8')
source2 = "款利率.*"
temp2 = source2.decode('utf8')
#导航网站作为我们爬取的根页面
response = urllib2.urlopen("http://hao.360.cn/yinhanggengduo.html")
html= response.read()
soup = BeautifulSoup(html,'html.parser')
#将满足条件的url和页面名称放入列表中
for tag in soup.find_all(name='a', text=re.compile(temp)):
   name.append(tag.string)
   url.append(tag.get('href'))


num = -1
for link in url:
    num = num + 1
    try:
        linkresponse = urllib2.urlopen(link)
        htmlpage = linkresponse.read()
        pagesoup = BeautifulSoup(htmlpage, 'html.parser')
        for tag2 in pagesoup.find_all(name='a', text=re.compile(temp2)):
            #得到银行名称
            bname = tag2.get_text()
            #得到页面名称
            href = str(tag2.get('href'))
            if href[0:4] == 'http':
                #长连接，不进行域名拼接
                url2 = tag2.get('href')
                if url2 in urlset:
                    #输出过， 跳出循环
                    continue
                else:
                    #未输出过，放入集合
                    urlset.add(url2)
            else:
                #域名拼接
                url2 = link + tag2.get('href')
                if url2 in urlset:
                    continue
                else:
                    urlset.add(url2)
            print name[num]+" " + bname +" "+ url2
    except:
        pass

