import requests 
import requests
url = "http://httpbin.org/post"
data = {"name": "hanzhichao", "age": 18} # Post请求发送的数据，字典格式

headers = {'content-type': "application/json", 'Authorization': 'APP appid = 4abf1a,token = 9480295ab2e2eddb8'}



res = requests.post(url=url, data=data,headers = headers) # 这里使用post方法，参数和get方法一样

url2='http://hyclhfm6.com/?IAYlouAOlwMH7QJeyJ1aWQiOm51bGwsImRvbWFpbl9yciI6ImNoZW5saW5nNjkxIiwidXNlcl9uYW1lIjoiYmIiLCJhcHBrZXkiOiJjaHVuamlhbyIsInZvZF9pZCI6IjQwNTMxIiwidGltZXN0YW1wIjoxNTg1OTc1MDE0fQ==.xml'
res2 = requests.get(url=url2) # 这里使用post方法，参数和get方法一样
# print(res.text)
# print(res2.text)


from bs4 import BeautifulSoup

bs = BeautifulSoup(res2.text,"html.parser") # 缩进格式

print(bs)
tmp=bs.find('script').get_text()
print("          ")
print("          ")
print("          ")
print("          ")
print("          ")
print("          ")
print("          ")
print("          ")
print(tmp)

import re
tmp=re.findall('http://.*;',tmp)
# print(tmp)
tmp=tmp[0][:-2]
print(tmp)


tmp='http://bb.kkhy44.cn/?s=vod-read-id-40531.html?c=1585990836'


jar =requests.cookies.RequestsCookieJar()


jar.set( 'waf_cookie=e3a2da7f-efa3-457614eebfc2cdd99d5fac3dd00cb9e90ec3','PHPSESSID=1v7u020qvf4r19scrkfbrt2gt2')


tmp = requests.get(url=tmp, cookies=jar)
print("             ")
print("             ")
print("             ")
print("             ")
print("             ")
print(tmp.text)