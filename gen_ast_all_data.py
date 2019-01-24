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
	return res

class InvalidToken(Exception):
	pass

data = []
vocab_src = set([])
vocab_dst = set([])

import pickle
with open('dataset_asm_ast/vocab_asm.pickle','rb') as fp:
	vocab_dst = pickle.load(fp)
with open('dataset_asm_csrc/vocab_csrc.pickle','rb') as fp:
	vocab_src = pickle.load(fp)

types = set([])

c_keywords = 	[
	'break', 'case','char','const', 'continue','default', 'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 
	'goto', 'if', 'inline','int','long','register','restrict', 'return','short', 'signed','sizeof','static','struct', 'switch','typedef', 
	'union', 'unsigned', 'void',  'while', 'volatile',
	'_Bool','__int128','_Complex',
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
	'setbe', 'bl', 'addsd', 'cmovge', 'divss', 'es', 'movs', 'fchs', 'rsi', 'cvtsi2sd', 'ror', 'cmovns', 'fmulp', 'andpd', 'r10d', 'setb', 'setne', '[', 'rcx', 'movaps', 'shr', 'cvttsd2si', 'xmm6', 'esi', 'cx', 'cmovg', 'cmovae', 'r15', 'movdqa', ':', 'and', 'setnp', 'fldcw', 'DWORD', 'fnstcw', ')', 'fldpi', 'fprem', 'al', 'WORD', '2', 'scas', 'sub', 'cvtpd2ps', 'or', 'ret', 'ja', 'mulsd', 'dil', 'r14b', 'r9d', 'mfence', 'QWORD', 'xmm0', 'ucomisd', 'fs', 'xmm2', 'fld', 'unpcklpd', '*', 'fstsw', 'si', 'sete', 'tzcnt', 'fdivp', 'dl', 'test', 'sahf', 'rax', '(', 'fsubp', 'sil', 'movd', 'cmp', 'r13', 'shrd', 'call', 'jp', 'bx', 'jg', 'add', '1', '8', 'subsd', 'divsd', 'setg', 'sqrtss', 'r12b', 'jae', 'leave', 'xmm5', 'imul', 'cmpxchg', 'cvttss2si', 'rsp', 'fsubrp', 'r11b', 'cdqe', 'jnp', 'movzx', 'cmova', 'ds', 'lea', 'andps', 'r9b', 'r15d', 'st', 'unpcklps', ',', 'idiv', 'xmm4', 'setp', 'movsx', 'TBYTE', 'mulss', 'js', 'repnz', 'setge', 'r8b', 'edi', 'stos', 'eax', 'xmm7', 'syscall', 'cdq', 'jle', 'fistp', 'sar', 'ah', 'rol', 'pop', 'ucomiss', 'push', 'jge', 'xmm1', 'movsd', 'xmm3', 'cmovbe', 'movapd', '0', 'addss', 'seta', 'cbw', 'subss', 'ecx', 'movabs', 'fldz', 'jl', 'int', 'cmovs', 'cvtps2pd', 'XMMWORD', ']', 'jne', 'xorpd', 'orpd', 'cmovl', 'lock', 'dx', 'r12w', 'bsf', 'jns', 'r8', 'jbe', 'nop', 'not', 'je', '-', 'xadd', 'pushf', 'movsxd', 'setae', 'xchg', 'pxor', 'r11', 'r12', 'xorps', 'r9', 'faddp', '+', 'neg', 'setle', 'di', 'fdivrp', 'jmp', 'r11d', 'r8d', 'r10b', 'rip', 'punpckldq', 'paddd', 'r12d', 'cwde', 'fild', 'cvtsi2ss', 'cmove', 'BYTE', 'ebx', 'edx', 'rep', 'fpatan', 'rbp', '4', 'r14d', 'ax', 'rdx', 'cqo', 'rbx', 'div', 'xor', 'pcmpeqd', 'mov', 'rdi', 'cmovle', 'fld1', 'r13b', 'fxch', 'bswap', 'psubd', 'mul', 'movq', 'cl', 'setl', 'shl', 'fabs', 'r13d', 'bsr', 'dh', 'fucomip', 'r10', 'jb', 'r14', 'movss', 'movmskpd', 'fstp','r9w','r8w']


import collections 
import os
import sys
import random
import asm_objd_conv
import csrc2ast
import my_c_ast


def getsub(fn,s):
	if fn[-len(s):]==s:
		return fn[:-len(s)]
	else:
		return None

os.chdir('./build_ast_save_csrc')
srcs = list(filter(lambda x: x is not None,map(lambda x: getsub(x,'.objd'),os.listdir('.'))))
#srcs = ['cat']


len_srcs = len(srcs)
idx_srcs = 0

file_start_num = 0
file_num_sum = 0
data_set_num = 0


def dump_save_data():
	global data,vocab_dst,file_start_num,file_num_sum,data_set_num
	data = list(map(lambda xy: (
						list(map(lambda t: vocab_dst.index(t),xy[0])),
						list(map(lambda t: vocab_src.index(t),xy[1]))
				),data))
	dfn = file_num_sum - file_start_num 
	dls = len(data)
	print('save idx:',data_set_num)
	print('file num diff:',dfn)
	print('data length:',dls)
	
	import pickle
	with open('../dataset_asm_csrc/data_%d.pickle' % data_set_num,'wb') as fp:
		pickle.dump((dfn,data),fp)
	
	file_start_num = file_num_sum
	data_set_num += 1
	data = []

for fn in srcs:
	print(fn)
	idx_srcs += 1
	if idx_srcs % 100 == 0:
		sys.stderr.write('finish %d/%d\n' % (idx_srcs,len_srcs))
	
	with open(fn + '.s') as fp:
		asm = fp.read()
	with open(fn + '.objd') as fp:
		objd = fp.read()
		if objd.count('\n')<=5:
			continue
	try:
		#print(asm,objd)
		#exit()
		ds = asm_objd_conv.get_line_from_list(asm,objd)
	except ParseErr as s:
		#print(s)
		continue

	#print(ds)
	with open(fn + '.tokenized.c') as fp:
		csrc = fp.read()
		csrc = csrc.split('\n')
	
	if len(csrc)>=10000: # too long.
		print('too long',fn)
		continue
	
	#os.system('clang -Xclang -dump-tokens -fsyntax-only ' + fn + '.c > /dev/null 2> ' + fn + '.tokenized')
	with open(fn + '.tokenized') as fp:
		csrc_with_type = list(map(lambda x: x.split(' ')[0],fp.read().split('\n')))
		#types |= set(csrc_with_type)


	with open(fn + '.parsed') as fp:
		parsed = fp.read()
	lines = parse_c(parsed)

	idents = []
	def i2srcdata(i):
		t = csrc_with_type[i]
		if t in ['continue', 'percentequal',  'while', 'pipeequal', 'minusminus',  'lesslessequal', 'double', 'l_paren', 'float',  'union', 'int', 'starequal', 'caretequal', 'l_brace', 'do', 'sizeof', 'lessequal', 'pipe', 'greatergreaterequal', 'break', 'question', 'comma', 'plusplus', 'r_paren', 'semi', 'long', 'struct', 'r_square', 'l_square',  'const', 'typeof',  'exclaimequal', 'static', 'register', 'greater',  'equal', 'tilde', 'eof', 'slash', 'minusequal', 'greaterequal', 'if', 'ampamp', 'amp', 'signed', 'unsigned', 'caret', 'typedef', 'lessless', 'greatergreater', 'ellipsis',  'restrict', 'switch', 'equalequal', 'period', 'else', 'extern',  'short', 'r_brace', 'colon', 'pipepipe', 'return', 'goto',  'default', 'enum', 'void', 'case', 'minus', 'plusequal', 'volatile', 'for', 'less', 'ampequal', 'exclaim', 'star', 'percent',  'arrow', 'slashequal', 'char', 'plus', 'inline','_Bool','__real','__imag','_Complex','__int128']:
			return csrc[i]
			#elif t in ['_Alignof','__alignof','_Static_assert', '__func__','__FUNCTION__','_Noreturn']: #C++かな。
			#	raise "Avoid"
			#elif t in ['__builtin_va_arg', '__attribute','__builtin_offsetof', '__extension__','__PRETTY_FUNCTION__','asm','__label__','wide_string_literal','__builtin_types_compatible_p']: #avoid like gcc extension.
			#	raise "Avoid"
		elif t in ['numeric_constant']:
			v = csrc[i]
			
			if "." in v or "p" in v or "P" in v or "e" in v or "E" in v:
				return "__DUMMY_floatliteral__"
			else:
				"""
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
				
				if tv[:2] in ['0x','0X']:
					d =  int(tv[2:],16)
				elif tv[:2] in ['0b','0B']:
					d =  int(tv[2:],2)
				else:
					d = int(tv)
				"""
				d = int(v)
				assert d>=0
				if d >= 256:
					d = 256
				return "%d" % d
		elif t in ['char_constant','wide_char_constant']:
			return "__DUMMY_charliteral__"
		elif t in ['string_literal']:
			return '__DUMMY_stringliteral__'
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
	try:
		ds = list(map(lambda i_d: (i_d[0],list(map(asm_trim,i_d[1]))),ds))
	except InvalidToken:
		continue
	"""
	for _,d in ds:
		vocab_dst |= set(filter(lambda x: x[:6] != 'LABEL_',d))
	vocab_dst |= set(['LA_%d' % i for i in range(LABEL_MAX+1)])
	vocab_dst |= set(['NEW_LINE'])
	maxlavellen = 0
	print(fn)
	continue
	"""
	
	for a,b in lines:
		if b-a > 200:
			continue # too big.
		#print(tree)
		"""
		print(' '.join(csrc[a-1:b]))
		last = my_c_ast.load_astdata(tree)
		last.before_show()
		print(last.show())
		"""
		
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
				#vocab_src |= set(tcs)
		
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
		except InvalidToken:
			pass
	
	#print(len(data))
	file_num_sum += 1
	
	dml = 100000
	if len(data)>=dml:
		mdata = data[dml:]
		data = data[:dml]
		dump_save_data()
		data = mdata

dump_save_data()

"""
vocab_src = list(vocab_src)
sys.stderr.write("vocab_src_len: %d\n" % len(vocab_src))
import pickle
with open('../dataset_asm_csrc/vocab_csrc.pickle','wb') as fp:
	pickle.dump(vocab_src,fp)
"""

#print(os.system('pwd'))
def show_data():
	for asm,tree in data:
		print('asm:',' '.join([vocab_dst[v] for v in asm]))
		nast = my_c_ast.load_astdata(tree)
		nast.before_show()
		print('source:',nast.show())

#show_data()










