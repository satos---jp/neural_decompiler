import os
import time

for i in range(1,35):
	print('curl',i)
	os.system('curl "https://api.github.com/search/repositories?q=topic:C&sort=stars&page=%d" | grep "html_url" > o.txt' % i)
	with open('o.txt') as fp:
		s = fp.read().split('\n')[1:-1]
	
	for d in s:
		d = d.split('"')[3]
		#print(d)
		#continue
		os.system('git clone --depth=1 %s' % d)
	#break
#curl -I "https://api.github.com/search/repositories?q=topic:C&sort=stars&page=%d"

