#!/usr/bin/python3

import os

#os.system('pwd')
#os.system('rm *')

import subprocess

import code
import binascii
import csrc2ast
import c_cfg


tmp_cnt = 0
def gen_tmp_name():
	global tmp_cnt
	tmp_cnt += 1
	return 'satos_temporary_struct_union_enum_decl_%d' % tmp_cnt

def chain_functions(fs):
	for f,ouf,errf in fs:
		d = subprocess.call(f,stdout=ouf,stderr=errf)
		#print(d,f)
		if d != 0:
			#exit()
			return False
	return True

devnull = open('/dev/null','w')
with open('correspond_table_ast.txt','r') as fp:
	corresp_table = fp.read()

def convert(path):
	global devnull
	global corresp_table
	if path in corresp_table:
		return
	bfn = subprocess.check_output(['sha1sum',path]).decode().split(' ')[0]
	#bfn = binascii.hexlify(path.encode()).decode()
	fn = bfn
	with open('../correspond_table_ast.txt','a') as fp:
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
		
		try:
			with open('%s.tokenized' % fn,"r") as fp:
				s = fp.read()
		except UnicodeDecodeError:
			print('mousiran',fn)
			return
		
		ts = ""
		bc = ""
		for cs in s.split('\n'):
			#print(cs)
			cs = "'".join(cs.split("'")[1:-1])
			if bc in ['struct','union','enum'] and cs == '{':
				ts += gen_tmp_name() + '\n'
			bc = cs
			ts += cs + '\n'
		with open('%s.tokenized.bef.c' % fn,"w") as fp:
			fp.write(ts)


	if d == 0:
		tkcfn = '%s.tokenized.c' % fn
		tkbefcfn = '%s.tokenized.bef.c' % fn
		if not chain_functions([(['gcc','-O0','-c',tkbefcfn],devnull,devnull)]):
			return
		print('fn:',tkbefcfn)
		try:
			psrc = csrc2ast.src2ast(tkbefcfn)
		except c_cfg.Oteage:
			print('oteage',fn)
			return
		except AssertionError:
			print('mousiran',fn)
			return
		except IndexError:
			print('mousiran',fn)
			return
		except ValueError:
			print('mousiran',fn)
			return
		with open(tkcfn,'w') as fp:
			fp.write(psrc)
		
		if not chain_functions([(['gcc','-O0','-c',tkcfn],devnull,devnull)]):
			print('mousiran',fn)
			return
	
	return
	
	
	"""
	satos@ispc2016:~/sotsuron/neural_decompiler/build_ast$ ls *.pp.c | wc
  	18109   18109  833014
	satos@ispc2016:~/sotsuron/neural_decompiler/build_ast$ ls *.tokenized | wc
		16829   16829  858279
	satos@ispc2016:~/sotsuron/neural_decompiler/build_ast$ ls *.tokenized.bef.c | wc
		16845   16845  960165
	satos@ispc2016:~/sotsuron/neural_decompiler/build_ast$ ls *.tokenized.bef.c.astdump | wc
		12728   12728  827320
	satos@ispc2016:~/sotsuron/neural_decompiler/build_ast$ ls *.tokenized.bef.o | wc
		12656   12656  721392
	satos@ispc2016:~/sotsuron/neural_decompiler/build_ast$ ls *.tokenized.c | wc
		10327   10327  547331
	satos@ispc2016:~/sotsuron/neural_decompiler/build_ast$ ls *.tokenized.o | wc
		 9615    9615  509595
	"""
	

	
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
	with open('../correspond_table_ast.txt','a') as fp:
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
#os.system('echo -n "" > correspond_table_ast.txt')
#os.system('rm build_ast/*')

os.chdir('./build_ast')
enumerate_src('../../corpus_folders/')

