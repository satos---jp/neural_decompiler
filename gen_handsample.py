import os
import csrc2ast
import asm_objd_conv
import my_c_ast
import pickle

with open('dataset_asm_ast/vocab_asm.pickle','rb') as fp:
	vocab_asm = pickle.load(fp)
with open('dataset_asm_csrc/vocab_csrc.pickle','rb') as fp:
	vocab_csrc = pickle.load(fp)

def getds(fn):
	os.system('clang -Xclang -ast-dump -fsyntax-only %s.c > %s.parsed 2>/dev/null' % (fn,fn))
	ast = csrc2ast.src2ast('%s.c' % fn,'%s.parsed' % fn)
	
	os.system('gcc -O0 -S -g -c %s.c -o %s.s' % (fn,fn))
	os.system('as %s.s -o %s.o' % (fn,fn))
	os.system('objdump -d -M intel -w %s.o > %s.objd' % (fn,fn))
	with open('%s.s' % fn,'r') as fp:
		asm = fp.read()
	with open('%s.objd' % fn,'r') as fp:
		objd = fp.read()
	ds = asm_objd_conv.get_line_from_list(asm,objd)
	astdata = ast.subexpr_line_list()
	
	
	res = []
	for (a,b),nsize,tree in astdata:
		nas = ''
		nas_d = []
		for i,d in ds:
			if a <= i and i < b:
				nas += ' '.join(d) + '\n'
				nas_d += d + ['NEW_LINE']
				#nas.append(d)
		if len(nas)==0:
			continue
		
		#print(nas)
		#print('.' * 100)
		csrc = csrc2ast.ast2token_seq(tree)
		#nast = my_c_ast.load_astdata(tree)
		#nast.before_show()
		#csrc = nast.show()
		#csrc = csrc.split(' ')
		
		def conv_c(x):
			if x[:7] == '__ALPHA':
				x = x.replace('__ALPHA_name_','__ID_')
			return x
		def conv_asm(x):
			if x[:6] == 'LABEL_':
				x = x.replace('LABEL_','LA_')
			return x
		csrc = list(map(conv_c,csrc))
		c_tks = list(map(lambda x: vocab_csrc.index(x),csrc))
		nas_d = list(map(conv_asm,nas_d))
		asm_tks = list(map(lambda x: vocab_asm.index(x),nas_d))
		#print(csrc)
		#print('-' * 100)
		res.append((asm_tks,c_tks,tree))
	
	return res

ds = getds('y')


print(len(ds))
with open('od.pickle','wb') as fp:
	pickle.dump(ds,fp)


