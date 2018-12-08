#!/usr/bin/python3

import os

os.chdir('build')
#os.system('pwd')
os.system('rm *')
os.system('cp ../../xv6-public/* ./')

import subprocess
fns = str(subprocess.check_output(['ls'])).split('\\n')
fns = filter(lambda x: x[-2:]=='.c',fns)
tfns = []
for fn in fns:
	fn = fn[:-2]
	d = subprocess.call('gcc -E %s.c | grep -v "#" | grep -v "^[[:space:]]*$" > %s.pp.c 2> /dev/null' % (fn,fn),shell=True)
	
	if d == 0:
		d = subprocess.call('clang -Xclang -dump-tokens -fsyntax-only %s.pp.c > /dev/null 2> %s.tokenized' % (fn,fn),shell=True)
	
	with open('%s.tokenized' % fn,"r") as fp:
		s = fp.read()
	
	ts = ""
	for cs in s.split('\n'):
		#print(cs)
		cs = "'".join(cs.split("'")[1:-1])
		ts += cs + '\n'
	
	with open('%s.tokenized.c' % fn,"w") as fp:
		fp.write(ts)
	
	if d == 0:
		d = subprocess.call('gcc -O0 -S -g -c %s.tokenized.c -o %s.s > /dev/null 2> /dev/null' % (fn,fn),shell=True)
	
	if d == 0:
		d = subprocess.call('as %s.s -o %s.o 2> /dev/null' % (fn,fn),shell=True)	

	if d == 0:
		d = subprocess.call('clang -Xclang -ast-dump -fsyntax-only %s.tokenized.c 2> /dev/null > %s.parsed' % (fn,fn),shell=True)

	
	if d == 0:
		subprocess.call('objdump -d -M intel -w %s.o > %s.objd' % (fn,fn),shell=True)
		tfns.append(fn)


print(tfns)
