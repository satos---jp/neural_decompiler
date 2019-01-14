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
	def debug(sl,x=None,ast_help=None):
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
		print(sl.kind)
		t,b = sl.top,sl.bottom
		print('source range:',t,b)
		with open(sl.filename) as fp:
			src = fp.readlines()[t-1:b]
		print(' '.join(src))
		code.interact(local={'sl':sl,'x': x,'cs':(list(x.get_children()) if x is not None else None),'ah':ast_help})
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
			sl.leafdata = x[1]
			sl.children = []
			sl.show = lambda: x[1]
			sl.isleaf = True
			sl.invalid = False
			return
		#sl.debug(x)r
		sl.isleaf = False
		sl.kind = x.kind
		sl.top = x.extent.start.line
		sl.bottom = x.extent.end.line
		sl.filename = x.extent.start.file.name
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
			#ast_hint.cs = ast_hint.cs[len(ast_hint.cs)-len(cvs):]
			ast_hint.cs = list(filter(lambda x: len(x.data)<=3 or x.data[3] != 'implicit',ast_hint.cs))
		elif sl.kind == CursorKind.VAR_DECL:
			assert ast_hint.tag == 'VarDecl'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: x.kind != CursorKind.TYPE_REF,cvs))
			cvs = cvs[len(cvs)-len(ast_hint.cs):] # for array size decl
		elif sl.kind == CursorKind.PARM_DECL:
			#sl.debug(x,ast_hint)
			#print(sl.top,sl.bottom,ast_hint.tag)
			assert ast_hint.tag == 'ParmVarDecl'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: x.kind != CursorKind.TYPE_REF,cvs))
			cvs = cvs[len(cvs)-len(ast_hint.cs):] # for array size decl
		elif sl.kind == CursorKind.FIELD_DECL:
			assert ast_hint.tag == 'FieldDecl'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: x.kind != CursorKind.TYPE_REF,cvs))
			cvs = cvs[len(cvs)-len(ast_hint.cs):] # for array size decl
		elif sl.kind == CursorKind.CXX_UNARY_EXPR:
			assert ast_hint.tag == 'UnaryExprOrTypeTraitExpr'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: x.kind != CursorKind.TYPE_REF,cvs))
			#cvs = cvs[len(cvs)-len(ast_hint.cs):]
			"""
			elif sl.kind == CursorKind.UNEXPOSED_EXPR and ast_hint.tag == 'VAArgExpr':
				sl.invalid = True
				return
			"""
		elif sl.kind == CursorKind.CSTYLE_CAST_EXPR:
			assert ast_hint.tag == 'CStyleCastExpr'
			#code.interact(local={'hi': ast_hint})
			cvs = list(filter(lambda x: not x.kind in [CursorKind.TYPE_REF,CursorKind.PARM_DECL],cvs))
			#cvs = cvs[len(cvs)-len(ast_hint.cs):]
		elif sl.kind == CursorKind.FUNCTION_DECL:
			assert ast_hint.tag == 'FunctionDecl'
			#code.interact(local={'hi': ast_hint})
			#cvs = list(filter(lambda x: not (x.kind == CursorKind.TYPE_REF or (x.kind == CursorKind.PARM_DECL and len(x.spelling) == 0)),cvs))
			cvs = list(filter(lambda x: not (x.kind in [
				CursorKind.TYPE_REF,CursorKind.UNEXPOSED_ATTR,CursorKind.PURE_ATTR,
				CursorKind.ASM_LABEL_ATTR,CursorKind.CONST_ATTR,
				
			] ),cvs))
			ast_hint.cs = list(filter(lambda x: x.tag[-4:] != 'Attr',ast_hint.cs))
			if x.spelling in [
				'execl','execle','execlp',
				'fprintf','printf','sprintf','snprintf','fscanf','scanf','sscanf'
			]:
				sl.invalid = True
			#print(len(cvs))
		elif sl.kind == CursorKind.GOTO_STMT:
			assert ast_hint.tag == 'GotoStmt'
			ast_hint.cs.append(clang_ast_dump.HelpAST('DUMMY'))
		
		hint_fcs = list(filter(lambda x: x is not None,ast_hint.cs))
		
		#print(sl.kind,ast_hint.tag)
		#print(list(map(lambda x: x.kind,cvs)),list(map(lambda x: x.data,hint_fcs)))
		
		if len(cvs) != len(hint_fcs):
			raise c_cfg.Oteage
			print('length mismatch')
			print(sl.kind,ast_hint.tag)
			print(list(map(lambda x: x.kind,cvs)),list(map(lambda x: x.data,hint_fcs)))
			sl.debug(x,ast_hint)
			assert False
		
		tcs = [AST(v,h,src) for v,h in zip(cvs,hint_fcs)]
		
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
			v = sl.checkalign([1])
			sl.show = lambda: v
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
			if len(na)==0 or ty in ['_LIB_VERSION_TYPE','struct _IO_FILE_plus']:
				sl.invalid = True
		elif sl.kind == CursorKind.STRUCT_DECL or sl.kind == CursorKind.UNION_DECL:
			na = x.spelling
			#if na == 'satos_temporary_struct_union_decl_16':
			#	sl.debug(x)
			assert len(na)>0
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
		elif sl.kind ==	CursorKind.CXX_UNARY_EXPR:
			tcs = [AST(('sizeof_kasou_var','8'))]

		tcs = list(filter(lambda a: not a.invalid,tcs))
		sl.children = tcs
		#print('inited',x.kind)
		
		if sl.kind in [CursorKind.STRUCT_DECL,CursorKind.UNION_DECL,CursorKind.ENUM_DECL]:
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
			
			try:
				if type(s) is str:
					#print(s,cs)
					res = s.format(*cs)
				elif type(s) is types.FunctionType:
					res = s(cs)
			except IndexError:
				print('indexerror')
				sl.debug()
			return res
		else:
			return 'INVALID'
	
	
	def treenize(sl):
		if type(sl.kind) is str:
			d = sl.leafdata
			if sl.kind in c_cfg.node_id_data.keys():
				c_cfg.node_id_data[sl.kind] |= set([d])
			else:
				c_cfg.node_id_data[sl.kind] = set([d])
		else:
			expt = c_cfg.cfg[sl.kind]
			for c in sl.children:
				c.treenize()
			
			#print(sl.kind)
			def f(v):
				nk = v.kind
				if type(nk) is not str:
					nk = c_cfg.reduced_type[nk]
				return nk
								
			cks = list(map(f,sl.children))
			#print(expt,cks)
			
			def same_type(expt,real):
				# expr can be stmt
				return expt == real or (expt,real) == ('stmt','expr')
				
			i = 0
			bexpt = expt
			
			try:
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
			except AssertionError:
				print('mousiran')
				raise AssertionError
				print('failure')
				print(sl.kind)
				print(bexpt,cks)
				sl.debug()
		
	def asdata(sl):
		pass
		# return for training data.
		# parent, nodetype, 
		
	def subexpr_line_list(sl):
		res = []
		for c in sl.children:
			res += c.subexpr_line_list()
		
		if not sl in [CursorKind.COMPOUND_STMT]:
			res.append((sl.top,sl.bottom),sl.asdata)
		return res

import sys

def src2ast(fn):
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

	try:
		ast_hint = clang_ast_dump.fn_to_clang_ast(fn)
	except clang_ast_dump.ParseErr:
		raise c_cfg.Oteage
	with open(fn) as fp:
		src = list(map(lambda x: x.strip(),fp.readlines()))
	index = Index.create()
	ast = index.parse(fn).cursor
	#sys.stdin.read()
	res = AST(ast,ast_hint,src)
	res.treenize()
	resrc = res.show()
	resrc = 'struct __va_list_tag{    unsigned int gp_offset;    unsigned int fp_offset;    void *overflow_arg_area;    void *reg_save_area;};' + resrc
	return resrc





fn = 'tesx'
if __name__ == '__main__':
	import os
	os.system('clang -Xclang -dump-tokens -fsyntax-only ' + fn + '.c 2>&1 |  sed -e "s/[^\']*\'\(.*\)\'[^\']*/\\1/" > ' + fn + '.tokenized.c')
	if os.system('gcc -S -std=c99 %s.c' % fn) != 0:
		print('%s.c uncompilable' %fn)
		exit()

	ass = src2ast('%s.tokenized.c' % fn)
	with open('t.c','w') as fp:
		fp.write(ass)
