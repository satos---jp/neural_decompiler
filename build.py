#!/usr/bin/python3

import os

#os.system('pwd')
#os.system('rm *')

import subprocess

import code
import binascii


def convert(path):
	bfn = subprocess.check_output(['sha1sum',path]).decode().split(' ')[0]
	bfn = bfn + '_' + binascii.hexlify(path.encode()).decode()
	if bfn in os.listdir('./build'):
		return bfn
	fn = './build/' + bfn
	subprocess.call('touch %s' % bfn,shell=True)

	d = subprocess.call('gcc -E %s > /dev/null 2> /dev/null' % path,shell=True) 
	
	if d == 0:
		d = subprocess.call('gcc -E %s | grep -v "#" | grep -v "^[[:space:]]*$" > %s.pp.c 2> /dev/null' % (path,fn),shell=True)
	
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
	
	if d == 0:
		d = subprocess.call('gcc -O0 -S -g -c %s.tokenized.c -o %s.s > /dev/null 2> /dev/null' % (fn,fn),shell=True)
	
	if d == 0:
		d = subprocess.call('as %s.s -o %s.o 2> /dev/null' % (fn,fn),shell=True)	

	if d == 0:
		d = subprocess.call('clang -Xclang -ast-dump -fsyntax-only %s.tokenized.c 2> /dev/null > %s.parsed' % (fn,fn),shell=True)

	
	if d == 0:
		d = subprocess.call('objdump -d -M intel -w %s.o > %s.objd' % (fn,fn),shell=True)
	
	if d == 0:
		print('convert',path,'to',fn)
		exit()
		return bfn		
	else:
		return None



def enumerate_src(path):
	print('search at',path)
	res = []
	for fn in os.listdir(path):
		rfn = os.path.join(path,fn)
		if rfn[-2:]=='.c':
			d = convert(rfn)
			if d is not None:
				res.append(d)
		elif os.path.isdir(rfn):
			res += enumerate_src(rfn)
	return res


"""
#for dirn in ['xv6-public/','git/']:
for dirn in ['nginx/src/%s/' % s for s in ['core','event','http','mail','stream']]:
	os.system('yes | rm -r src/')
	os.system('cp -r ../../corpus_folders/%s src' % dirn)
	fns = subprocess.check_output(['ls','./src/']).decode().split('\\n')
	#code.interact(local={'fns':fns})
	fns = filter(lambda x: x[-2:]=='.c',fns)
	tfns = []
"""


tfns = enumerate_src('../corpus_folders/')

print(tfns)
