#! /usr/bin/env /home/satos/sotsuron/neural_decompiler/python_fix_seed.sh

class InvalidToken(Exception):
	pass

data = []
vocab_src = set([])
vocab_dst = set([])

import pickle
with open('dataset_asm_ast/vocab_asm.pickle','rb') as fp:
	vocab_dst = pickle.load(fp)

import c_cfg
with open('dataset_asm_ast/c_trans_array.pickle','wb') as fp:
	pickle.dump(c_cfg.trans_array,fp)

types = set([])

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
						xy[1],
						xy[2]
				),data))
	dfn = file_num_sum - file_start_num 
	dls = len(data)
	print('save idx:',data_set_num)
	print('file num diff:',dfn)
	print('data length:',dls)
	
	import pickle
	with open('../dataset_asm_ast/data_%d.pickle' % data_set_num,'wb') as fp:
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
	
	try:
		astdata = csrc2ast.src2ast(fn + '.tokenized.c',fn + '.parsed').subexpr_line_list()
	except c_cfg.Oteage:
		continue
	
	for (a,b),nsize,tree in astdata:
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
			# data augumentation
			aug_tcs = []
			for _ in range(3):
				aug_tcs.append(c_cfg.alphaconv(tree))
			
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
					data.append((a,nsize,b))
		except InvalidToken:
			pass

	file_num_sum += 1
	
	dml = 1000000
	if len(data)>=dml:
		mdata = data[dml:]
		data = data[:dml]
		dump_save_data()
		data = mdata

dump_save_data()


"""
vocab_dst = list(vocab_dst)
sys.stderr.write("vocab_dst_len: %d\n" % len(vocab_dst))

import pickle
with open('../dataset_asm_ast/vocab_asm.pickle','wb') as fp:
	pickle.dump(vocab_dst,fp)
"""

#print(os.system('pwd'))
def show_data():
	for asm,tree in data:
		print('asm:',' '.join([vocab_dst[v] for v in asm]))
		nast = my_c_ast.load_astdata(tree)
		nast.before_show()
		print('source:',nast.show())

#show_data()










