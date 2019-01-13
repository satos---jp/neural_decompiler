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


struct_decl_list = []
import copy
class AST:
	def debug(sl,x):
		
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
		global struct_decl_list;
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
		
		#print(ast_hint)
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
		elif sl.kind == CursorKind.FIELD_DECL:
			assert ast_hint.tag == 'FieldDecl'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: x.kind != CursorKind.TYPE_REF,cvs))
			cvs = cvs[len(cvs)-len(ast_hint.cs):]
		elif sl.kind == CursorKind.FUNCTION_DECL:
			assert ast_hint.tag == 'FunctionDecl'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: not (x.kind == CursorKind.TYPE_REF or (x.kind == CursorKind.PARM_DECL and len(x.spelling) == 0)),cvs))
			#print(len(cvs))
		elif sl.kind == CursorKind.GOTO_STMT:
			assert ast_hint.tag == 'GotoStmt'
			ast_hint.cs.append(clang_ast_dump.HelpAST('DUMMY'))
		
		hint_fcs = list(filter(lambda x: x is not None,ast_hint.cs))
		
		#print(sl.kind,ast_hint.tag)
		#print(list(map(lambda x: x.kind,cvs)),list(map(lambda x: x.data,hint_fcs)))
		
		if len(cvs) != len(hint_fcs):
			sl.debug(x)
			assert False
		
		tcs = [AST(v,h,src) for v,h in zip(cvs,hint_fcs)]
		tcs = list(filter(lambda a: not a.invalid,tcs))
		
		#print(tcs)
		
		if sl.kind == CursorKind.BINARY_OPERATOR or sl.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
			assert len(tcs)==2
			lc,rc = tcs
			#assert lc.top == sl.top and rc.bottom == sl.bottom and lc.bottom+2==rc.top
			op = sl.checkalign([lc,1,rc])
			if sl.kind == CursorKind.BINARY_OPERATOR:
				opty = 'binop'
			elif sl.kind == CursorKind.COMPOUND_ASSIGNMENT_OPERATOR:
				opty = 'casop'
			tcs = [lc,AST((opty,op)),rc]
		if sl.kind == CursorKind.UNARY_OPERATOR:
			#sl.debug(x)
			assert len(tcs)==1
			rc = tcs[0]
			#assert lc.top == sl.top and rc.bottom == sl.bottom and lc.bottom+2==rc.top
			if 'prefix' in ast_hint.data:
				op = sl.checkalign([1,rc])
				isprefix = True
			else:
				op = sl.checkalign([rc,1])
				isprefix = False
			tcs = [AST(('unop',(op,isprefix))),rc]
		elif sl.kind == CursorKind.INTEGER_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = x.literal
			sl.show = lambda: str(v)
			tcs = [AST(('integerliteral',v))]
		elif sl.kind == CursorKind.CHARACTER_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = x.literal
			sl.show = lambda: "'"+str(v)+"'"
			tcs = [AST(('charliteral',v))]
			#sl.debug(x)
		elif sl.kind == CursorKind.STRING_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = src[sl.top-1]
			sl.show = lambda: str(v)
			tcs = [AST(('stringliteral',v))]
		elif sl.kind == CursorKind.FLOATING_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = src[sl.top-1]
			sl.show = lambda: str(v)
			tcs = [AST(('floatliteral',v))]
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
			tcs = [AST(('name',la))]
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
		elif sl.kind == CursorKind.VAR_DECL or sl.kind == CursorKind.FIELD_DECL:
			na = x.spelling
			ty = x.type.get_canonical().spelling
			tcs = [AST(('type',ty)),AST(('name',na)),] + tcs
		elif sl.kind == CursorKind.STRUCT_DECL or sl.kind == CursorKind.UNION_DECL:
			#sl.debug(x)
			na = x.spelling
			tcs = [AST(('name',na)),] + tcs
		elif sl.kind == CursorKind.FUNCTION_DECL:
			#sl.debug(x)
			na = x.spelling
			ty = x.type.get_canonical().spelling
			tcs = [AST(('type',ty)),AST(('name',na)),] + tcs
		elif sl.kind == CursorKind.PARM_DECL:
			#sl.debug(x)
			na = x.spelling
			ty = x.type.get_canonical().spelling
			tcs = [AST(('type',ty)),AST(('name',na)),]
			#print(tcs,ty,na)
		elif sl.kind == CursorKind.CSTYLE_CAST_EXPR:
			ty = x.type.get_canonical().spelling
			tcs = [AST(('type',ty))] + tcs
		elif sl.kind ==	CursorKind.MEMBER_REF_EXPR:
			#sl.debug(x)
			assert len(tcs)==1
			re = tcs[0]
			op,na = sl.checkalign([re,1,1])
			assert na == x.spelling
			tcs = [re,AST(('memberrefop',op)),AST(('name',na))]
		elif sl.kind ==	CursorKind.ENUM_DECL:
			na = x.spelling
			tcs = [AST(('name',na))] + tcs
		elif sl.kind ==	CursorKind.ENUM_CONSTANT_DECL:
			na = x.spelling
			tcs = [AST(('name',na))] + tcs

		sl.children = tcs
		#print('inited',x.kind)
		
		if sl.kind == CursorKind.STRUCT_DECL or sl.kind == CursorKind.UNION_DECL:
			struct_decl_list.append(copy.copy(sl))
			sl.invalid = True
		elif sl.kind == CursorKind.TRANSLATION_UNIT:
			sl.children = struct_decl_list + sl.children
			struct_decl_list = []
	
	def show(sl):
		#print(sl,sl.kind,sl.invalid)
		if not sl.invalid:
			cs = list(map(lambda v: v.show(),sl.children))
			res = str(sl.kind) + '()' #+ ','.join(cs) + ')'
			s = c_cfg.cfg2str[sl.kind]
			
			if type(s) is str:
				#print(s,cs)
				res = s.format(*cs)
			elif type(s) is types.FunctionType:
				res = s(cs)
			return res
		else:
			return 'INVALID'
	
	
	def treenize(sl):
		if type(sl.kind) is str:
			pass
		else:
			expt = c_cfg.cfg[sl.kind]
			for c in sl.children:
				c.treenize()
			
			print(sl.kind)
			def f(v):
				nk = v.kind
				if type(nk) is not str:
					nk = c_cfg.reduced_type[nk]
				return nk
								
			cks = list(map(f,sl.children))
			print(expt,cks)
			
			def same_type(expt,real):
				# expr can be stmt
				return expt == real or (expt,real) == ('stmt','expr')
				
			i = 0
			while i < len(cks):
				nk = cks[i]
				if type(expt[0]) is list:
					if expt[0][1]=='list':
						if same_type(expt[0][0],nk):
							pass
						else:
							expt = expt[1:]
							continue
					elif expt[0][1]=='option':
						if same_type(expt[0][0],nk) or nk == 'NULL':
							expt = expt[1:]
						else:
							expt = expt[1:]
							continue
					else:
						assert Failure
				else:
					et = expt[0]
					assert same_type(et,nk)
					expt = expt[1:]
				i += 1
			
			assert all(map(lambda x: x[1] in ['list','option'],expt))
		
	def asdata(sl):
		# return for training data.
		# parent, nodetype, 
		
	def subexpr_line_list(sl):
		res = []
		for c in sl.children:
			res += c.subexpr_line_list()
		
		if not sl in [CursorKind.COMPOUND_STMT]:
			res.append((sl.top,sl.bottom),sl.asdata())
		return res

import sys

def src2ast(fn):
	ast_hint = clang_ast_dump.fn_to_clang_ast(fn)
	with open(fn) as fp:
		src = list(map(lambda x: x.strip(),fp.readlines()))
	index = Index.create()
	ast = index.parse(fn).cursor
	#sys.stdin.read()
	res = AST(ast,ast_hint,src)
	res.treenize()
	return res.show()



import os
os.system('clang -Xclang -dump-tokens -fsyntax-only tesx.c 2>&1 |  sed -e "s/[^\']*\'\(.*\)\'[^\']*/\\1/" > tesx.tokenized.c')
if os.system('gcc -S -std=c99 tesx.c') != 0:
	print('tesx.c uncompilable')
	exit()

if __name__ == '__main__':
	ass = src2ast('tesx.tokenized.c')
	with open('t.c','w') as fp:
		fp.write(ass)
