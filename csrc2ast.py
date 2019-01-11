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

class AST:
	def debug(sl,x):
		"""
		vs = dir(x)
		def f(a):
			try:
				res = x.__getattribute__(a)
			except AssertionError:
				res = AssertionError
			res = (a,res)
			return res
		
		ats = list(map(f,vs))
		for kv in ats:
			print(kv)
		"""
		code.interact(local={'sl':sl,'x': x,'cs':list(x.get_children())})
		exit()
	
	def checkalign(sl,vs):
		h = sl.top
		res = []
		for d in vs:
			if d == 1:
				res.append(sl.src[h-1])
				h += d
			elif type(d) is str:
				#print(sl.src[h-1],d)
				assert sl.src[h-1]==d
				h += 1
			else:
				assert h == d.top
				h = d.bottom + 1
		assert h == sl.bottom + 1
		
		if len(res)==1:
			res = res[0]
		return res
	
	def __init__(sl,x,ast_hint=None,src=None):
		sl.src = src
		if type(x) is tuple:
			sl.kind = x[0]
			sl.children = []
			sl.show = lambda: x[1]
			sl.isleaf = True
			return
		#sl.debug(x)r
		sl.isleaf = False
		sl.kind = x.kind
		sl.top = x.extent.start.line
		sl.bottom = x.extent.end.line
		#print(x.kind,sl.top,sl.bottom)
		d = c_cfg.cfg[sl.kind]
		sl.invalid = d is None
		if sl.invalid:
			return
		
		print(ast_hint)
		cvs = list(x.get_children())
		if sl.kind == CursorKind.TRANSLATION_UNIT:
			#print(ast_hint.tag)
			assert ast_hint.tag == 'TranslationUnitDecl'
			ast_hint.cs = ast_hint.cs[len(ast_hint.cs)-len(cvs):]
		elif sl.kind == CursorKind.VAR_DECL:
			assert ast_hint.tag == 'VarDecl'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: x.kind != CursorKind.TYPE_REF,cvs))
			cvs = cvs[len(cvs)-len(ast_hint.cs):]
		elif sl.kind == CursorKind.FUNCTION_DECL:
			assert ast_hint.tag == 'FunctionDecl'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: x.kind != CursorKind.TYPE_REF,cvs))
		elif sl.kind == CursorKind.GOTO_STMT:
			assert ast_hint.tag == 'GotoStmt'
			ast_hint.cs.append(clang_ast_dump.HelpAST('DUMMY'))
		
		hint_fcs = list(filter(lambda x: x is not None,ast_hint.cs))
		print(sl.kind,ast_hint.tag)
		print(cvs,hint_fcs)
		
		if len(cvs) != len(hint_fcs):
			sl.debug(x)
			assert False
		
		tcs = [AST(v,h,src) for v,h in zip(cvs,hint_fcs)]
		
		print(tcs)
		
		if sl.kind == CursorKind.BINARY_OPERATOR or sl.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
			assert len(tcs)==2
			lc,rc = tcs
			#assert lc.top == sl.top and rc.bottom == sl.bottom and lc.bottom+2==rc.top
			op = sl.checkalign([lc,1,rc])
			tcs = [lc,AST(('binop',op)),rc]
		if sl.kind == CursorKind.UNARY_OPERATOR:
			#sl.debug(x)
			assert len(tcs)==1
			rc = tcs[0]
			#assert lc.top == sl.top and rc.bottom == sl.bottom and lc.bottom+2==rc.top
			if 'prefix' in ast_hint.data:
				op = sl.checkalign([1,rc])
				tcs = [AST(('unop',op)),rc]
			else:
				op = sl.checkalign([rc,1])
				tcs = [rc,AST(('unop',op))]
		elif sl.kind == CursorKind.INTEGER_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = x.literal
			sl.show = lambda: str(v)
		elif sl.kind == CursorKind.CHARACTER_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = x.literal
			sl.show = lambda: "'"+str(v)+"'"
			#sl.debug(x)
		elif sl.kind == CursorKind.STRING_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = src[sl.top-1]
			sl.show = lambda: str(v)
			#sl.debug(x)
		elif sl.kind == CursorKind.LABEL_STMT:
			assert len(tcs)==1
			s = tcs[0]
			la = sl.checkalign([1,':',s])
			tcs = [AST(('label',la)),s]
		elif sl.kind == CursorKind.LABEL_REF:
			assert len(tcs)==0
			la = sl.checkalign([1])
			tcs = [AST(('label',la))]
		elif sl.kind == CursorKind.DECL_REF_EXPR:
			assert len(tcs)==0
			la = sl.checkalign([1])
			tcs = [AST(('label',la))]
		elif sl.kind == CursorKind.FOR_STMT:
			assert len(ast_hint.cs)==5 and ast_hint.cs[1] is None
			mcs = tcs
			tcs = []
			for a in [ast_hint.cs[0],ast_hint.cs[2],ast_hint.cs[3]]:
				if a is None:
					tcs.append(AST(('NULL','')))
				else:
					tcs.append(mcs[0])
					mcs = mcs[1:]
			tcs.append(mcs[0])
		elif sl.kind == CursorKind.VAR_DECL:
			na = x.spelling
			ty = x.type.get_canonical().spelling
			tcs = [AST(('type',ty)),AST(('name',na)),] + tcs
		elif sl.kind == CursorKind.FUNCTION_DECL:
			#sl.debug(x)
			na = x.spelling
			ty = x.type.get_canonical().spelling
			tcs = [AST(('type',ty)),AST(('name',na)),] + tcs
		
		sl.children = tcs
		#print('inited',x.kind)
		
	
	def show(sl):
		print(sl,sl.kind,sl.invalid)
		if not sl.invalid:
			cs = list(map(lambda v: v.show(),sl.children))
			res = str(sl.kind) + '(' + ','.join(cs) + ')'
			s = c_cfg.cfg2str[sl.kind]
			
			if type(s) is str:
				print(s,cs)
				res = s.format(*cs)
			elif type(s) is types.FunctionType:
				res = s(cs)
			return res
		else:
			return 'INVALID'
	

import sys

def src2ast(fn):
	ast_hint = clang_ast_dump.fn_to_clang_ast(fn)
	with open(fn) as fp:
		src = list(map(lambda x: x.strip(),fp.readlines()))
	index = Index.create()
	ast = index.parse(fn).cursor
	#sys.stdin.read()
	res = AST(ast,ast_hint,src)
	print(res.show())



import os
os.system('clang -Xclang -dump-tokens -fsyntax-only tesx.c 2>&1 |  sed -e "s/[^\']*\'\(.*\)\'[^\']*/\\1/" > tesx.tokenized.c')
if os.system('gcc -S -std=c99 tesx.c') != 0:
	print('tesx.c uncompilable')
	exit()

if __name__ == '__main__':
	src2ast('tesx.tokenized.c')
	
