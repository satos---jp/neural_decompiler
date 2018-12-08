#!/usr/bin/python3

	
from pycparser import c_parser, c_ast, parse_file
	
srcs = ['bio', 'bootmain', 'cat', 'console', 'echo', 'exec', 'file', 'forktest', 'fs', 'grep', 'ide', 'init', 'ioapic', 'kalloc', 'kbd', 'kill', 'lapic', 'ln', 'ls', 'main', 'memide', 'mkdir', 'mkfs', 'mp', 'picirq', 'pipe', 'printf', 'rm', 'sh', 'sleeplock', 'stressfs', 'string', 'syscall', 'sysfile', 'sysproc', 'uart', 'ulib', 'umalloc', 'wc', 'zombie']


class ParseErr(Exception):
	def __init__(sl,s):
		pass

def s2tree(s):
	s = s.split('\n')
	ls = len(s)
	i = 0
	def parse(d):
		nonlocal i,s,ls
		
		node = s[i][d:]
		nt = node.split(' ')[0]
		#print(node)
		#try:
		ps = re.findall('<line:\d*:\d, line:\d*:\d*>',node)
		if len(ps)>0:
			pos = ps[0]
		else:
			pos = None
		
		node = (nt,pos)
		
		i += 1
		#print(i,d)
		#print(s[i])
		if i >= ls or len(s[i])<=d or (s[i][d]!='|' and s[i][d]!='`'):
			#print(node)
			return (node,[])
		
		res = []
		while True:
			if s[i][d:d+2]=='|-':
				res.append(parse(d+2))
			elif s[i][d:d+2]=='`-':
				res.append(parse(d+2))
				break
			else:
				#return res
				#print(s[i][d:])
				raise ParseErr('miss at %d:%d' % (i,d))
		return (node,res)
	
	return parse(0)

import code
import re

def parse_c(prsc):
	#print(s2tree(prsc))
	
	res = []
	tree = s2tree(prsc)
	#print(tree)
	def trav(d):
		nonlocal res
		ntp,ds = d
		nt,pos = ntp
		
		# CompoundStmt は多分飛ばさないといけない(if文のあととかがへんになる)
		if nt != 'CompoundStmt' and pos is not None:
			v = pos.split(':')
			a,b = int(v[1])-1,int(v[3])-1 #0-indexed !!
			#print(nt,pos,a,b)
			res.append((a,b))
		
		for x in ds:
			trav(x)
		
	trav(tree)
	
	"""
	lines = re.findall('<line:\d*:\d*, line:\d*:\d*>',prsc)
	#print(lines)
	#code.interact(local={'tree':tree})
	
	res = []
	for d in lines:
		v = d.split(':')
		#print(v)
		a,b = int(v[1])-1,int(v[3])-1 #0-indexed !!
		#print(a,b)
		res.append((a,b))
	"""
	
	return res
	
def get_line_from_list(asm,objd):
	asm = asm.split('\n')
	asm2 = filter(lambda x: x!='' and x[0]!='.' and x[:2] != '\t.',asm)
	asm2 = list(asm2)
	asm = filter(lambda x: x!='' and x[0]!='.' and (x[:2] != '\t.' or x[:5]=='\t.loc'),asm)
	asm = list(asm)
	
	objd = objd.split('\n')[5:]
	objd = filter(lambda x: x!='',objd)
	objd = list(objd)
	
	#print(len(asm2),len(objd))
	#print('\n'.join(asm2))
	#print('-' * 100)
	#print('\n'.join(objd))
	#assert len(objd) == len(asm2)
	if len(objd) != len(asm2):
		raise ParseErr("nanka hen")
	
	"""
	print(len(asm),len(objd))
	print('\n'.join(asm))
	print('-' * 100)
	print('\n'.join(objd))
	"""
	
	i = 0
	nl = ""
	res = []
	for s in asm:
		if s[:5]=='\t.loc':
			nl = s
		else:
			if s[-1]!=':':
				nld = nl.split(' ')[2]
				#print(nld,s,objd[i])
				#res.append((nld,objd[i]))
				
				#print(nld,' ',s)
				#print(nld,' ',objd[i].split('\t')[1])
				nbs = objd[i].split('\t')[1]
				nbs = filter(lambda x: x!='',nbs.split(' '))
				nbs = list(nbs)
				#print(nld,nbs)
				res.append((int(nld)-1,nbs)) #1-indexed!!
			i += 1
	return res


data = []
#srcs = ['cat']
for fn in srcs:
	fn = './build/' + fn
	with open(fn + '.s') as fp:
		asm = fp.read()
	with open(fn + '.objd') as fp:
		objd = fp.read()
	try:
		ds = get_line_from_list(asm,objd)
	except ParseErr as s:
		#print(s)
		continue
	#print(ds)
	with open(fn + '.parsed') as fp:
		parsed = fp.read()
	#print(fn)
	lines = parse_c(parsed)
	#print(ds)
	#print(lines)
	#continue
	
	with open(fn + '.tokenized.c') as fp:
		csrc = fp.read()
		csrc = csrc.split('\n')
	
	#print(fn)
	for a,b in lines:
		nas = []
		for i,d in ds:
			if a <= i and i < b:
				nas += d
				#nas += 
		if len(nas)==0:
			continue
		
		ncs = csrc[a:b+1] 
		#print(nas)
		#print(ncs)
		
		data.append((nas,ncs))

print('data = ' + str(data))





