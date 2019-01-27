import os
import csrc2ast
import asm_objd_conv
import my_c_ast

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
	
	for (a,b),nsize,tree in astdata:
		nas = ''
		for i,d in ds:
			if a <= i and i < b:
				nas += ' '.join(d) + '\n'
				#nas.append(d)
		if len(nas)==0:
			continue
		
		print(nas)
		print('.' * 100)
		nast = my_c_ast.load_astdata(tree)
		nast.before_show()
		print(nast.show())
		print('-' * 100)



getds('y')
