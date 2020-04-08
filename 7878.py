import requests


tmp='http://bb.kkhy44.cn/?s=vod-read-id-40531.html?c=1585990836'

headers = {'Content-Security-Policy': 'upgrade-insecure-requests'}
jar =requests.cookies.RequestsCookieJar()


jar.set( 'waf_cookie','e3a2da7f-efa3-457614eebfc2cdd99d5fac3dd00cb9e90ec3')
jar.set( 'PHPSESSID','1v7u020qvf4r19scrkfbrt2gt2')


tmp = requests.get(url=tmp, cookies=jar,headers = headers)

print(tmp.text)