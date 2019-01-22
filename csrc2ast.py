#! /usr/bin/env /home/satos/sotsuron/neural_decompiler/python_fix_seed.sh
#coding: utf-8

import c_cfg
import clang_ast_dump
import types

import code

from clang.cindex import Config,Index,CursorKind
Config.set_library_path("/usr/lib/llvm-7/lib")



# clang python bindings is inefficient,
# so I need to use information from clang -ast-dump 



import tokenizer
import my_c_ast

def ast2token_seq(data):
	ast = my_c_ast.load_astdata(data)
	ast.before_show()
	#code.interact(local={'ast':ast})
	src = ast.show()
	with open('tmp.c','w') as fp:
		fp.write(src)
	#print(src)
	res = tokenizer.tokenize('tmp.c')
	#print(src,res)
	return res

import sys

def src2ast(fn,pfn):
	if fn in [
		'b50b97a5b3e07dd44019949b5fb6ed2303edf46a.tokenized.bef.c', # null case
		'4c77c60d44f4fa40e75d0ac10a9c1e76f4fa9843.tokenized.bef.c', # null exposedexpr
		'0606d3a176fa430a5f41a083cd555378dcc499b5.tokenized.bef.c', # strformat
		'fn: 598d7a9f44792c035d4eda119765daa378bc710f.tokenized.bef.c', #null label
		'598d7a9f44792c035d4eda119765daa378bc710f.tokenized.bef.c', #null label
		'7c76f2a14cc170cf7c4e99e77d5ea907a342bfd1.tokenized.bef.c', # NULL file
		'463844b04dc5a35e393ad9e862aff9e20784f667.tokenized.bef.c', #hen
		'faa4d26aa28d2681ae5d47c85009011ca5bc1047.tokenized.bef.c',
		'bd817f210aff59c4c7f32d02f74236cd5a9747f7.tokenized.bef.c', #misterious syntax like void (*signal(int sig, void (*func)(int)))(int)
		'a669457110bb0f34b581e2ecd4531215c8c0340c.tokenized.bef.c'
		'83316377e3c0aa84edbdfab83251a88ff0b0e134.tokenized.bef.c', #file structure
		'557fddcfd8b5b2ace68901180fe715867f98b0fd.tokenized.bef.c',
		'94401323ace568313410a8e199b2b87112b54ccf.tokenized.bef.c', #enum cast
		'390537b683a13d85c512ab0f5fce6ab3eaaefc37.tokenized.bef.c', # non type sizeof
		'3cad1ef0ef40208856c167cb8aeffc491bafdea5.tokenized.bef.c',
		'17d67d5ed4c95dbb7a48abc807e1d4620c94d7b2.tokenized.bef.c', #nazo
		'cd31b577d5dc633f211d80b941a7e35974260217.tokenized.c', #ioctl
		
		'c124c78dd9cf922a6b12de292eda02174e44149b.tokenized.bef.c', # TODO(satos) nannka type emmbedding bug
		'28bb854f96b9f83734ef199d0da2982d0905ff38.tokenized.bef.c',
	]:
		raise c_cfg.Oteage
	
	c_cfg.init_idxnize()
	try:
		ast_hint = clang_ast_dump.fn_to_clang_ast(pfn)
	except clang_ast_dump.ParseErr:
		raise c_cfg.Oteage
	with open(fn) as fp:
		src = list(map(lambda x: x.strip(),fp.readlines()))
	index = Index.create()
	ast = index.parse(fn).cursor
	#sys.stdin.read()
	res = my_c_ast.AST(ast,ast_hint,src)
	res.get_astdata()
	#resrc = res.show()
	#resrc = 'struct __va_list_tag{    unsigned int gp_offset;    unsigned int fp_offset;    void *overflow_arg_area;    void *reg_save_area;};' + resrc
	return res


import random


def randtree(ty):
	cs = c_cfg.trans_array[ty]
	nc = random.randint(0,len(cs)-(1 if random.randint(0,10) else 0))
	
	if nc == len(cs):
		return (ty,-1,[])
	else:
		res = [randtree(d) for d in cs[nc]]
		return (ty,nc,res)

def fuzzing():
	for _ in range(10000):
		tree = randtree(c_cfg.rootidx)
		if len(tree[2])==0:
			continue
		#print(tree)
		ts = ast2token_seq(tree)
		#assert not ts is None
		#print(ts)

"""
print(list(enumerate(c_cfg.alltypes)))
[(0, ['expr', 'option']), (1, ['stmt', 'option']), (2, ['decl', 'list']), (3, ['expr', 'list']), (4, ['stmt', 'list']), (5, ['tyspec', 'list']), (6, ['paramlist', 'option']), (7, ['integerliteral', 'option']), (8, ['typename', 'list']), (9, 'integerliteral'), (10, 'stringliteral'), (11, 'floatliteral'), (12, 'charliteral'), (13, 'binop'), (14, 'unop'), (15, 'casop'), (16, 'memberrefop'), (17, 'NULL'), (18, 'tyspec'), (19, 'name'), (20, 'structname'), (21, 'unionname'), (22, 'enumname'), (23, 'label'), (24, 'toplevel'), (25, 'labelref'), (26, 'expr'), (27, 'stmt'), (28, 'decl'), (29, 'type_type'), (30, 'paramlist'), (31, 'type'), (32, 'typename'), (33, 'typedecltype'), (34, CursorKind.ARRAY_SUBSCRIPT_EXPR), (35, CursorKind.BINARY_OPERATOR), (36, CursorKind.BREAK_STMT), (37, CursorKind.CALL_EXPR), (38, CursorKind.CASE_STMT), (39, CursorKind.CHARACTER_LITERAL), (40, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR), (41, CursorKind.COMPOUND_STMT), (42, CursorKind.CONDITIONAL_OPERATOR), (43, CursorKind.CONTINUE_STMT), (44, CursorKind.CSTYLE_CAST_EXPR), (45, CursorKind.CXX_UNARY_EXPR), (46, CursorKind.DECL_REF_EXPR), (47, CursorKind.DECL_STMT), (48, CursorKind.DEFAULT_STMT), (49, CursorKind.DO_STMT), (50, CursorKind.ENUM_CONSTANT_DECL), (51, CursorKind.ENUM_DECL), (52, CursorKind.FIELD_DECL), (53, CursorKind.FLOATING_LITERAL), (54, CursorKind.FOR_STMT), (55, CursorKind.FUNCTION_DECL), (56, CursorKind.GOTO_STMT), (57, CursorKind.IF_STMT), (58, CursorKind.INIT_LIST_EXPR), (59, CursorKind.INTEGER_LITERAL), (60, CursorKind.LABEL_REF), (61, CursorKind.LABEL_STMT), (62, CursorKind.MEMBER_REF_EXPR), (63, CursorKind.NULL_STMT), (64, CursorKind.PAREN_EXPR), (65, CursorKind.PARM_DECL), (66, CursorKind.RETURN_STMT), (67, CursorKind.STRING_LITERAL), (68, CursorKind.STRUCT_DECL), (69, CursorKind.SWITCH_STMT), (70, CursorKind.TRANSLATION_UNIT), (71, CursorKind.UNARY_OPERATOR), (72, CursorKind.UNEXPOSED_EXPR), (73, CursorKind.UNION_DECL), (74, CursorKind.VAR_DECL), (75, CursorKind.WHILE_STMT), (76, <class 'pycparser.c_ast.Typename'>), (77, <class 'pycparser.c_ast.ArrayDecl'>), (78, <class 'pycparser.c_ast.TypeDecl'>), (79, <class 'pycparser.c_ast.PtrDecl'>), (80, <class 'pycparser.c_ast.FuncDecl'>), (81, <class 'pycparser.c_ast.ParamList'>), (82, <class 'pycparser.c_ast.Struct'>), (83, <class 'pycparser.c_ast.Union'>), (84, <class 'pycparser.c_ast.Enum'>), (85, <class 'pycparser.c_ast.IdentifierType'>)]


[[[], [26]], [[], [27]], [[], [28, 2]], [[], [26, 3]], [[], [27, 4]], [[], [18, 5]], [[], [30]], [[], [9]], [[], [32, 8]], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[]], [[]], [[]], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], []], [[], []], [[]], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []], [[70]], [[60]], [[34], [35], [37], [39], [40], [42], [44], [45], [46], [53], [58], [59], [62], [64], [67], [71], [72], [67]], [[36], [38], [41], [43], [47], [48], [49], [54], [56], [57], [61], [63], [66], [69], [75], [75], [34], [35], [37], [39], [40], [42], [44], [45], [46], [53], [58], [59], [62], [64], [67], [71], [72], [67]], [[50], [51], [52], [65], [68], [73], [74], [55]], [[77], [80], [78], [79]], [[81]], [[76]], [[76]], [[82], [83], [84], [85]], [[26, 26]], [[26, 13, 26]], [[]], [[26, 3]], [[26, 27]], [[12]], [[26, 15, 26]], [[4]], [[26, 26, 26]], [[]], [[31, 26]], [[]], [[19]], [[2]], [[27]], [[27, 26]], [[19, 0]], [[22, 2]], [[31, 19, 0]], [[11]], [[1, 0, 0, 27]], [[31, 19, 2, 1]], [[25]], [[26, 27, 1]], [[3]], [[9]], [[23]], [[23, 27]], [[26, 16, 19]], [[]], [[26]], [[31, 19]], [[0]], [[10]], [[20, 2]], [[26, 27]], [[2]], [[14, 26]], [[26]], [[21, 2]], [[31, 19, 0]], [[26, 27]], [[5, 29]], [[7, 29]], [[5, 33]], [[5, 29]], [[6, 29]], [[8]], [[20]], [[21]], [[22]], [[5]], [[34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47], [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59], [60], [61], [62], [63], [64], [65], [66], [67], [68], [69], [70], [71], [72], [73], [74], [75]]]

"""

#print(c_cfg.s2tree_with_parenthes('union __ALPHA_unionname_30__ ()[]'))
#exit()
#fuzzing()
#exit()

def unflatten(v,ty=c_cfg.rootidx):
	#print(v,ty)
	if len(v)==0:
		return (ty,-1,[])
	_,ch = v[0]
	v = v[1:]
	cts = self.trans_data[ty][ch]
	#print(cts)
	rcs = []
	for ct in cts:
		ac,v = unflatten(v,ct)
		rcs.append(ac)
	
	return (ty,ch,rcs),v



fn = 'p'
if __name__ == '__main__':
	import os
	os.system('clang -Xclang -dump-tokens -fsyntax-only ' + fn + '.c 2>&1 |  sed -e "s/[^\']*\'\(.*\)\'[^\']*/\\1/" > ' + fn + '.tokenized.c')
	if os.system('gcc -S -std=c99 %s.tokenized.c' % fn) != 0:
		print('%s.tokenized.c uncompilable' %fn)
		exit()
	
	os.system('clang -Xclang -ast-dump -fsyntax-only %s.tokenized.c > %s.astdump 2>/dev/null' % (fn,fn))
	csrc = open('%s.tokenized.c' % fn,'r').read().split('\n')
	ast = src2ast('%s.tokenized.c' % fn,'%s.astdump' % fn)
	aststr = ast.show()
	#print(aststr)
	#exit()
	with open('t.c','w') as fp:
		fp.write(aststr)
	#print(ast.get_astdata())
	#c_cfg.data_miyasui(ast.get_astdata())
	c_cfg.validate_astdata(ast.get_astdata(),CursorKind.TRANSLATION_UNIT)
	print('validate passed')
	ds = ast.subexpr_line_list()
	#print(ds)
	#print(ds[-1])
	
	for (a,b),nsize,tree in ds:
		#if nsize<5:
		#	continue
		print(tree)
		print(' '.join(csrc[a-1:b]))
		print(a,b)
		nast = my_c_ast.load_astdata(tree)
		nast.before_show()
		print(nast.show())
		
		tree = c_cfg.alphaconv(tree)
		print(tree)
		nast = my_c_ast.load_astdata(tree)
		nast.before_show()
		print(nast.show())
	
	
