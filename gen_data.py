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
	asm = list(map(lambda x: x.split('#')[0].strip(),asm.split('\n')))
	asm_with_loc = list(filter(lambda x: x!='' and x[0]!='#' and x[-1]!=':' and (x[0] != '.' or x[:5]=='.loc '),asm))
	asm_noloc = list(filter(lambda x: x!='' and x[0]!='#' and x[-1]!=':' and x[0]!='.',asm))

	"""
	while asm_with_loc[-1][-1]==':':
		asm_with_loc = asm_with_loc[:-1]
	while asm_noloc[-1][-1]==':':
		asm_noloc = asm_noloc[:-1]
	"""
	#exit()
	objd = objd.split('\n')[5:]
	objd = list(filter(lambda x: x!='',objd))
	objd = list(map(lambda x: x.split('#')[0].strip(),objd))
	objd_asm_len = len(list(filter(lambda x: x[-1]!=':',objd)))
	"""
	# only for clang
	tobjd = []
	for i,v in enumerate(objd[:-1]):
		if objd[i+1][-1]==':':
			continue
		tobjd.append(v)
	objd = tobjd + [objd[-1]]
	"""
	#print(objd)
	#print(len(asm2),len(objd))
	#assert len(objd) == len(asm2)
	#print(len(objd),len(asm_noloc))
	"""
	print('\n'.join(asm_with_loc))
	print('-' * 100)
	print('\n'.join(objd))
	"""
	if objd_asm_len != len(asm_noloc):
		"""
		print('\n'.join(asm_noloc))
		print('-' * 100)
		print('\n'.join(objd))
		print('nanka hen',objd_asm_len,len(asm_noloc))
		exit()
		
		asm volatile があるので諦める
		"""
		raise ParseErr("nanka hen")
	
	"""
	print(len(asm),len(objd))
	print('\n'.join(asm_with_loc))
	print('-' * 100)
	print('\n'.join(objd))
	"""
	
	addrs = set([])
	for d in objd:
		if d[-1]==':':
			continue
		op = d.split('\t')[2]
		if op[0]=='j' or op[:4] == 'call':
			nad = list(filter(lambda a: a!='',op.split(' ')))[1]
			addrs |= set([nad])
			#print(op,nad)
	
	addrs = list(addrs)
	tobjd = []
	for d in objd:
		if d[-1]==':':
			if d[:-1] in addrs:
				tobjd.append(['LABEL_%d' % addrs.index(d[:-1]),':'])
			continue
		naddr = d.split('\t')[0][:-1].strip()
		if naddr in addrs:
			tobjd.append(['LABEL_%d' % addrs.index(naddr),':'])
		op = d.split('\t')[2]
		if op[0]=='j' or op[:4] == 'call':
			nad = list(filter(lambda a: a!='',op.split(' ')))[1]
			tobjd.append([op.split(' ')[0],'LABEL_%d' % addrs.index(nad)])
		else:
			nops = []
			be = ''
			op += ' '
			for c in op:
				if c.isalnum():
					be += c
				else:
					if be != '':
						if be != 'PTR':
							nops.append(be)
						be = ''
					if c != ' ':
						nops.append(c)
			
			tobjd.append(nops)
	#exit()
	
	i = 0
	res = []
	nld = 0
	for s in asm_with_loc:
		if s[:5]=='.loc ':
			# nld = s.split('\t')[1].split(' ')[1] # for clang -S
			nld = s.split(' ')[2] # for gcc -S
		else:
			#print(nld,tobjd[i])
			while tobjd[i][-1]==':':
				res.append((int(nld)-1,tobjd[i])) #1-indexed!!
				i += 1
			res.append((int(nld)-1,tobjd[i])) #1-indexed!!
			i += 1
	return res


# do-simasyo
# alignof 

# nokeru?
# restrict register volatile

# hihyouzyun
# typeof
c_keywords = 	[
	'break', 'case','char','const', 'continue','default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 
	'goto', 'if', 'inline','int','long','register','restrict', 'return','short', 'signed','sizeof','static','struct', 'switch','typedef', 
	'union', 'unsigned', 'void',  'while', 'volatile'
]

# ellipsis ...   (fmt,...)
c_operator_symbls = [
	'period', 'star',  'arrow','question', 'colon', 'semi', 'ellipsis', 'eof',  
	'plus', 'minus', 'slash','percent', 'lessless','minusminus', 'plusplus', 'greatergreater',
	'equalequal', 'ampamp', 'amp', 'pipepipe', 'pipe', 'caret', 'exclaim','comma', 'less', 'tilde', 'greater', 
	'equal', 'percentequal', 'pipeequal', 'lesslessequal','minusequal','plusequal',  'greaterequal',  'greatergreaterequal', 'ampequal',  'starequal', 'caretequal', 'lessequal', 'slashequal', 'exclaimequal', 
	'r_paren', 'l_paren', 'l_brace', 'r_brace', 'l_square', 'r_square', 
	 'typeof',  
]

x64_opcodes_registers = [
	'setbe', 'bl', 'addsd', 'cmovge', 'divss', 'es', 'movs', 'fchs', 'rsi', 'cvtsi2sd', 'ror', 'cmovns', 'fmulp', 'andpd', 'r10d', 'setb', 'setne', '[', 'rcx', 'movaps', 'shr', 'cvttsd2si', 'xmm6', 'esi', 'cx', 'cmovg', 'cmovae', 'r15', 'movdqa', ':', 'and', 'setnp', 'fldcw', 'DWORD', 'fnstcw', ')', 'fldpi', 'fprem', 'al', 'WORD', '2', 'scas', 'sub', 'cvtpd2ps', 'or', 'ret', 'ja', 'mulsd', 'dil', 'r14b', 'r9d', 'mfence', 'QWORD', 'xmm0', 'ucomisd', 'fs', 'xmm2', 'fld', 'unpcklpd', '*', 'fstsw', 'si', 'sete', 'tzcnt', 'fdivp', 'dl', 'test', 'sahf', 'rax', '(', 'fsubp', 'sil', 'movd', 'cmp', 'r13', 'shrd', 'call', 'jp', 'bx', 'jg', 'add', '1', '8', 'subsd', 'divsd', 'setg', 'sqrtss', 'r12b', 'jae', 'leave', 'xmm5', 'imul', 'cmpxchg', 'cvttss2si', 'rsp', 'fsubrp', 'r11b', 'cdqe', 'jnp', 'movzx', 'cmova', 'ds', 'lea', 'andps', 'r9b', 'r15d', 'st', 'unpcklps', ',', 'idiv', 'xmm4', 'setp', 'movsx', 'TBYTE', 'mulss', 'js', 'repnz', 'setge', 'r8b', 'edi', 'stos', 'eax', 'xmm7', 'syscall', 'cdq', 'jle', 'fistp', 'sar', 'ah', 'rol', 'pop', 'ucomiss', 'push', 'jge', 'xmm1', 'movsd', 'xmm3', 'cmovbe', 'movapd', '0', 'addss', 'seta', 'cbw', 'subss', 'ecx', 'movabs', 'fldz', 'jl', 'int', 'cmovs', 'cvtps2pd', 'XMMWORD', ']', 'jne', 'xorpd', 'orpd', 'cmovl', 'lock', 'dx', 'r12w', 'bsf', 'jns', 'r8', 'jbe', 'nop', 'not', 'je', '-', 'xadd', 'pushf', 'movsxd', 'setae', 'xchg', 'pxor', 'r11', 'r12', 'xorps', 'r9', 'faddp', '+', 'neg', 'setle', 'di', 'fdivrp', 'jmp', 'r11d', 'r8d', 'r10b', 'rip', 'punpckldq', 'paddd', 'r12d', 'cwde', 'fild', 'cvtsi2ss', 'cmove', 'BYTE', 'ebx', 'edx', 'rep', 'fpatan', 'rbp', '4', 'r14d', 'ax', 'rdx', 'cqo', 'rbx', 'div', 'xor', 'pcmpeqd', 'mov', 'rdi', 'cmovle', 'fld1', 'r13b', 'fxch', 'bswap', 'psubd', 'mul', 'movq', 'cl', 'setl', 'shl', 'fabs', 'r13d', 'bsr', 'dh', 'fucomip', 'r10', 'jb', 'r14', 'movss', 'movmskpd', 'fstp']

"""
['0x12345678', '0x8000000', '0x8abc0000', '0x400000', '0x6000beef', '0xfffffff', '0xffffff9d', '0x7fffff', '0xffffff80', '0x40000000', '0xffffffffffffffff', '0x5bd1e995', '0xffffffff', '0xff7fffff', '0x7ff0000000000000', '0xffffffe9', '0x80000000', '0x4024000000000000', '0x7fffffff', '0x8000000000000000', '0x20000000', '0x432b4544434241', '0x800000', '0x200000', '0xff9fffff', '0x10000000000', '0x7f800000', '0x7f7fffff', '0xfffffc03', '0x10000000', '0x807fffff', '0x1000000', '0x1869c', '0x7fa00000', '0xff800000']
"""




class InvalidToken(Exception):
	pass

data = []
vocab_src = set([])
vocab_dst = set([])
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
	print(fn)
	idx_srcs += 1
	if idx_srcs % 10 == 0:
		sys.stderr.write('finish %d/%d\n' % (idx_srcs,len_srcs))
	
	fn = './build/' + fn
	#sys.stderr.write(fn + '\n')
	with open(fn + '.s') as fp:
		asm = fp.read()
	with open(fn + '.objd') as fp:
		objd = fp.read()
		if objd.count('\n')<=5:
			continue
	try:
		#print(asm,objd)
		#exit()
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
		elif t in ['__real','__imag','_Alignof','__alignof','_Static_assert','_Bool', '__func__','__FUNCTION__','_Noreturn']: #C++かな。
			#raise "Avoid"
			pass
		elif t in ['__builtin_va_arg', '__attribute','__builtin_offsetof', '__extension__','_Complex','__PRETTY_FUNCTION__','asm','__label__','wide_string_literal','__builtin_types_compatible_p']: #avoid like gcc extension.
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
			if isinstance(d,int) and abs(d) <= 256:
				return "%d" % d #一種の正規化。しといたほうがいいはず。
			else:
				return "__CONST__"
		elif t in ['char_constant','wide_char_constant']:
			return csrc[i]
		elif t in ['string_literal']:
			return '__STR__'
		elif t in ['identifier']:
			tvn = '__VAR__' + csrc[i]
			idents.append(tvn)
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
	
	nibeki = [1<<(i*8) for i in range(8)]
	def asm_trim(v):
		if v in x64_opcodes_registers:
			return v
		elif v[:6]=='LABEL_':
			return v
		elif v[:2]=='0x':
			d = int(v[2:],0x10)
			if d <= 256 or ((d + 256) & 0x1ffffffffffffff00) in nibeki:
				return v
			else:
				return '__HEX__'
		else:
			print('unknown asm token:',v,'in file',fn)
		raise InvalidToken
			
				
	
	LABEL_MAX = 60
	ds = list(map(lambda i_d: (i_d[0],list(map(asm_trim,i_d[1]))),ds))
	for _,d in ds:
		vocab_dst |= set(filter(lambda x: x[:6] != 'LABEL_',d))
	vocab_dst |= set(['LA_%d' % i for i in range(LABEL_MAX+1)])
	vocab_dst |= set(['NEW_LINE'])
	maxlavellen = 0
	print(fn)
	for a,b in lines:
		nas = []
		for i,d in ds:
			if a <= i and i < b:
				nas += d + ['NEW_LINE']
				#nas.append(d)  
		if len(nas)==0:
			continue
		ilabels = list(set(list(filter(lambda x: x[:6]=='LABEL_',nas))))
		
		try:
			idents = []
			ncs = [i2srcdata(i) for i in range(a,b+1)]
			
			if len(ncs)>=100:
				raise InvalidToken # 一旦飛ばす。 ./build/c60873877f8813ca522babe077f477816cf9d124 とかがやばい。
				# 1000だと、 ./build/a578572061c49b5429968f8644ca95508a0473ac とかもやばい。
			idents = list(set(idents))
			
			# data augumentation
			
			aug_tcs = []
			for _ in range(3):
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
				aug_tcs.append(tcs)
				vocab_src |= set(tcs)
			
			aug_asm = []
			for _ in range(3):
				nds = []
				for d in ilabels:
					while True:
						v = random.randint(0,LABEL_MAX)
						if v in nds:
							continue
						nds.append(v)
						break
				#print(ncs,idents)
				
				tas = [("LA_%d" % nds[ilabels.index(c)] if c in ilabels else c) for c in nas] 
				aug_asm.append(tas)
			
			#maxlavellen = max(maxlavellen,
			
			for a in aug_asm:
				for b in aug_tcs:
					data.append((a,b))
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
vocab_dst = list(vocab_dst)
sys.stderr.write("vocab_dst_len: %d\n" % len(vocab_dst))
#print(vocab_dst)

#vocab_dst = list(filter(lambda x: x[:6]!='LABEL_' and x[:2]!='0x',vocab_dst))
#print(vocab_dst)
#exit()

data = list(map(lambda xy: (
					list(map(lambda t: vocab_dst.index(t),xy[0])),
					list(map(lambda t: vocab_src.index(t),xy[1]))
				),data))

#data = list(map(lambda xy: (list(map(lambda t: int(t,16),xy[0])),list(map(lambda t: vocab_src.index(t) if t in vocab_src else lvoc,xy[1]))),data))
#data = list(map(lambda xy: (list(map(lambda t: t,xy[0])),list(map(lambda t: vocab_src.index(t) if t in vocab_src else lvoc,xy[1]))),data))
#vocab_src.append('__SOME__')

#data = list(map(lambda xy: (list(map(lambda t: int(t,16),xy[0])),xy[1]),data))


#print('data = ' + str(data))
#print('c_vocab = ' + str(vocab_src))  
print('data length',len(data))

import pickle
with open('data.pickle','wb') as fp:
	pickle.dump((data,vocab_src,vocab_dst),fp)

def show_data():
	for k,vs in data:
		print('data:',k)
		print('source:',' '.join([vocab_src[v] for v in vs]))

#show_data()










