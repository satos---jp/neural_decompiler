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

def ast2token_seq(data,terminal_ids):
	ast = c_ast.AST([],isnull=True).load_astdata(data,terminal_ids)
	src = ast.show()
	with open('tmp','w') as fp:
		fp.write(src)
	return tokenizer.tokenize('tmp')

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





fn = 'y'
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
	
	
