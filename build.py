#!/usr/bin/python3

import os

#os.system('pwd')
#os.system('rm *')

import subprocess

import code
import binascii


def chain_functions(fs):
	for f,ouf,errf in fs:
		d = subprocess.call(f,stdout=ouf,stderr=errf)
		#print(d,f)
		if d != 0:
			#exit()
			return False
	return True

devnull = open('/dev/null','w')
with open('correspond_table.txt','r') as fp:
	corresp_table = fp.read()
	
def convert(path):
	global devnull
	global corresp_table
	if path in corresp_table:
		return
	bfn = subprocess.check_output(['sha1sum',path]).decode().split(' ')[0]
	#bfn = binascii.hexlify(path.encode()).decode()
	fn = bfn
	with open('../correspond_table.txt','a') as fp:
		fp.write('%s | %s\n' % (path,bfn))
	
	print(path)
	d = subprocess.call('clang -E %s > /dev/null 2> /dev/null' % path,shell=True) 
	
	if d == 0:
		d = subprocess.call('clang -E %s | grep -v "#" | grep -v "^[[:space:]]*$" > %s.pp.c 2> /dev/null' % (path,fn),shell=True)

	if d == 0 and int(list(filter(lambda x: x!= '',subprocess.check_output(['wc','%s.pp.c' % fn]).decode().split(' ')))[0]) > 5000:
		# too big.
		d = 1
	
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
	
	#print('tokenized',path)
	tcfn = '%s.tokenized.c' % fn
	sfn = '%s.s' % fn
	ofn = '%s.o' % fn
	
	with open('%s.parsed' % fn,"w") as parsedfp:
		with open('%s.objd' % fn,"w") as objdfp:
			isok = chain_functions([
				(['gcc','-O0','-S','-g','-c',tcfn,'-o',sfn]        ,devnull,devnull),
				(['as',sfn,'-o',ofn]                                 ,devnull,devnull),
				(['clang','-Xclang','-ast-dump','-fsyntax-only',tcfn],parsedfp,devnull),
				(['objdump','-d','-M','intel','-w',ofn]              ,objdfp,devnull),
			])
	
	parsedfp.close()
	objdfp.close()
	
	if isok:
		print('convert',path,'to',fn)
	else:
		#pass
		os.system('rm %s.*' % fn)





def enumerate_src(path):
	global table
	print('search at',path)
	mems = '%s | SEARCHED\n' % path
	if mems in corresp_table:
		return
	
	for fn in os.listdir(path):
		rfn = os.path.join(path,fn)
		if rfn[-2:]=='.c' and (not os.path.isdir(rfn)):
			convert(rfn)
		if os.path.isdir(rfn):
			enumerate_src(rfn)
	with open('../correspond_table.txt','a') as fp:
		fp.write(mems)

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

import os
os.chdir('./build')
enumerate_src('../../corpus_folders/')

