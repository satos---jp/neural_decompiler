import subprocess

devnull = open('/dev/null','w')
def tokenize(fn):
	global devnull
	assert fn[-2:]=='.c'
	p = subprocess.Popen(['clang','-Xclang','-dump-tokens','-fsyntax-only',fn],stdout=devnull,stderr=subprocess.PIPE)
	_,s = p.communicate()
	s = s.decode()
	#print(s)
	res = 0
	if len(s)>0:
		ts = []
		bc = ""
		for cs in s.split('\n'):
			#print(cs)
			cs = "'".join(cs.split("'")[1:-1])
			ts.append(cs)
		return ts
	else:
		return None	

