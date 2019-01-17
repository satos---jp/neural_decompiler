import os

def hassub(fn,s):
	return fn[len(fn)-len(s):]==s

fs = os.listdir('build_ast')

tos = list(filter(lambda fn: hassub(fn,'.tokenized.o'),fs))
print(len(tos))
tos = set(tos)
pcs = list(filter(lambda fn: hassub(fn,'.pp.c'),fs))
print(len(pcs))

i = 0
for fn in pcs:
	fn = fn.split('.')[0]
	if fn + '.tokenized.o' in tos:
		continue
	"""
	for ext in ['.pp.c','.tokenized','.tokenized.bef.c','.tokenized.bef.c.astdump','.tokenized.bef.o','.tokenized.c']:
		try:
			os.remove('./build_ast/' + fn + ext)
		except FileNotFoundError:
			pass
	"""
	#os.system('rm ./build_ast/' + fn + '.*')
	i += 1
	if i % 100 == 0:
		print(i)
print(i,36762-17932)

