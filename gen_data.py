#! /usr/bin/env /home/satos/sotsuron/neural_decompiler/python_fix_seed.sh

from pycparser import c_parser, c_ast, parse_file
	
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
				#nbs = objd[i].split('\t')[2]
				
				res.append((int(nld)-1,nbs)) #1-indexed!!
			i += 1
	return res


class InvalidToken(Exception):
	pass

data = []
vocab_src = set([])
types = set([])

import collections 
import os
import sys
import random

srcs = list(map(lambda x: x[:-5],filter(lambda x: x[-5:]=='.objd',os.listdir('./build'))))
#srcs = ['cat']

len_srcs = len(srcs)
idx_srcs = 0
for fn in srcs:
	idx_srcs += 1
	if idx_srcs % 100 == 0:
		sys.stderr.write('finish %d/%d\n' % (idx_srcs,len_srcs))
	
	fn = './build/' + fn
	#sys.stderr.write(fn + '\n')
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

	with open(fn + '.tokenized') as fp:
		csrc_with_type = list(map(lambda x: x.split(' ')[0],fp.read().split('\n')))
		#types |= set(csrc_with_type)
	
	idents = []
	def i2srcdata(i):
		t = csrc_with_type[i]
		if t in ['continue', 'percentequal',  'while', 'pipeequal', 'minusminus',  'lesslessequal', 'double', 'l_paren', 'float',  'union', 'int', 'starequal', 'caretequal', 'l_brace', 'do', 'sizeof', 'lessequal', 'pipe', 'greatergreaterequal', 'break', 'question', 'comma', 'plusplus', 'r_paren', 'semi', 'long', 'struct', 'r_square', 'l_square',  'const', 'typeof',  'exclaimequal', 'static', 'register', 'greater',  'equal', 'tilde', 'eof', 'slash', 'minusequal', 'greaterequal', 'if', 'ampamp', 'amp', 'signed', 'unsigned', 'caret', 'typedef', 'lessless', 'greatergreater', 'ellipsis',  'restrict', 'switch', 'equalequal', 'period', 'else', 'extern',  'short', 'r_brace', 'colon', 'pipepipe', 'return', 'goto',  'default', 'enum', 'void', 'case', 'minus', 'plusequal', 'volatile', 'for', 'less', 'ampequal', 'exclaim', 'star', 'percent',  'arrow', 'slashequal', 'char', 'plus', 'inline']:
			return csrc[i]
		elif t in ['__real','__imag','_Alignof','__alignof','_Static_assert','_Bool', '__func__','_Noreturn']: #C++かな。
			#raise "Avoid"
			pass
		elif t in ['__builtin_va_arg', '__attribute','__builtin_offsetof', '__extension__','_Complex','__PRETTY_FUNCTION__','asm','__label__','__builtin_types_compatible_p']: #avoid like gcc extension.
			pass
		elif t in ['numeric_constant']:
			# TODO ここできればうまくやりたい(ないとvocavが6435とかいわれてきつい)
			v = csrc[i]
			if "." in v or "p" in v or "e" in v or "E" in v:
				# 0x1p64 なんか、こういう記法もあるらしい。
				# 面倒なので、全部CONSTにする
				"""
				tv = v
				if tv[-1:] in ['f','L']:
					tv = tv[:-1]
				if re.match("\d*.\d*e\d*",tv):
					d = eval(tv)
				else:
					d = float(tv)
				"""
				return "__CONST__"
			else:
				tv = v
				if tv[-3:] in ['ULL','LLU','ull','uLL','llU','LLu']:
					tv = tv[:-3]
				elif tv[-2:] in ['UL','LU', 'LL','ul','ll']:
					tv = tv[:-2]
				elif tv[-1:] in ['L','l','U','u']:
					tv = tv[:-1]
				
				if tv[-1] not in '1234567890abcdefABCDEF':
					sys.stderr.write('bad integer representation: ' + v + '\n')
					exit()
				
				if tv[:2]=='0x':
					d =  int(tv[2:],16)
				elif tv[:2] in ['0b','0B']:
					d =  int(tv[2:],2)
				else:
					d = int(tv)
			if isinstance(d,int) and abs(d) <= 100:
				return "%d" % d #一種の正規化。しといたほうがいいはず。
			else:
				return "__CONST__"
		elif t in ['char_constant','wide_char_constant']:
			return csrc[i]
		elif t in ['string_literal']:
			return '__STR__'
		elif t in ['identifier']:
			idents.append('__VAR__' + csrc[i])
			return '__VAR__' + csrc[i]
		elif t in ['']:
			pass
		else:
			print('unknown token:',t,'in file',fn)
		raise InvalidToken
		#return csrc[i]
	
	for i in range(len(csrc_with_type)):
		try:
			i2srcdata(i)
		except InvalidToken:
			pass
	
	#print(fn)
	for a,b in lines:
		nas = []
		for i,d in ds:
			if a <= i and i < b:
				nas += d
				#nas.append(d)  
		if len(nas)==0:
			continue
		
		try:
			idents = []
			ncs = [i2srcdata(i) for i in range(a,b+1)]
			
			if len(ncs)>=100:
				raise InvalidToken # 一旦飛ばす。 ./build/c60873877f8813ca522babe077f477816cf9d124 とかがやばい。
				# 1000だと、 ./build/a578572061c49b5429968f8644ca95508a0473ac とかもやばい。
			idents = list(set(idents))
			
			# data augumentation
			for _ in range(5):
				nds = []
				for d in idents:
					while True:
						v = random.randint(0,60) #VAR_MAX 
						if v in nds:
							continue
						nds.append(v)
						break
				#print(ncs,idents)
				tcs = [("__ID_%d__" % nds[idents.index(c)] if c in idents else c) for c in ncs] 
				data.append((nas,tcs))
				vocab_src |= set(tcs)
						
			#vocab_src += ncs
		
			#data.append((nas,ncs))
		except InvalidToken:
			pass




"""
./build/0848b888848180a1ae75b12bede3681371189593.pp.c:9:19: warning: trigraph ignored [-Wtrigraphs]
  char *s8 = "??\\??=";                 ^
"""
#print(list(types))
#exit()

#vocab_src = sorted(map(lambda xy: (xy[1],xy[0]),collections.Counter(vocab_src).items()))[::-1][:1000]
#lvoc = len(vocab_src)
#vocab_src = list(map(lambda xy: xy[1],vocab_src))
vocab_src = list(vocab_src)
sys.stderr.write("vocab_src_len: %d\n" % len(vocab_src))
#print(vocab_src)
#exit()
data = list(map(lambda xy: (list(map(lambda t: int(t,16),xy[0])),list(map(lambda t: vocab_src.index(t) if t in vocab_src else lvoc,xy[1]))),data))
#data = list(map(lambda xy: (list(map(lambda t: t,xy[0])),list(map(lambda t: vocab_src.index(t) if t in vocab_src else lvoc,xy[1]))),data))
#vocab_src.append('__SOME__')

#data = list(map(lambda xy: (list(map(lambda t: int(t,16),xy[0])),xy[1]),data))


#print('data = ' + str(data))
#print('c_vocab = ' + str(vocab_src))  

import pickle
with open('data.pickle','wb') as fp:
	pickle.dump((data,vocab_src),fp)

def show_data():
	for k,vs in data:
		print('data:',k)
		print('source:',' '.join([vocab_src[v] for v in vs]))

#show_data()










