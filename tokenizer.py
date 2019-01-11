

def conv(fn):
	if d == 0:
		d = subprocess.call('clang -Xclang -dump-tokens -fsyntax-only %s.pp.c > /dev/null 2> %s.tokenized' % (fn,fn),shell=True)
	
		with open('%s.tokenized' % fn,"r") as fp:
			s = fp.read()
		
		if '__asm__' in s: #避けたい...
			d = -1
		
		ts = ""
		for cs in s.split('\n'):
			#print(cs)
			cs = "'".join(cs.split("'")[1:-1])
			ts += cs + '\n'
	
		with open('%s.tokenized.c' % fn,"w") as fp:
			fp.write(ts)
