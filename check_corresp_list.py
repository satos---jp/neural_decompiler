with open('correspond_table.txt','r') as fp:
	s = fp.read().split('\n')[1:-1]


hs = [d.split('/')[2] for d in s]
hs = set(hs)
print(len(list(hs)))
	
