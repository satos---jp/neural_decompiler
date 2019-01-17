import c_cfg
import clang_ast_dump
import types
from clang.cindex import Config,Index,CursorKind

import code

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
	
	def __init__(sl,x,ast_hint=None,src=None,*,isnull=False):
		if isnull:
			sl.kind = None
			return
		global struct_decl_list
		sl.src = src
		if type(x) is tuple:
			sl.kind = x[0]
			sl.leafdata = x[1]
			sl.children = []
			sl.show = lambda: sl.leafdata
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
			struct_decl_list = []
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
			v = str(x.literal)
			tcs = [AST(('integerliteral',v))]
		elif sl.kind == CursorKind.CHARACTER_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = sl.checkalign([1])
			tcs = [AST(('charliteral',v))]
		elif sl.kind == CursorKind.STRING_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = sl.checkalign([1])
			tcs = [AST(('stringliteral',v))]
		elif sl.kind == CursorKind.FLOATING_LITERAL:
			assert len(tcs)==0
			sl.isleaf = True
			v = str(x.literal)
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
			
			tree_branch = [[]]
			
			try:
				while i < len(cks):
					nk = cks[i]
					nch = sl.children[i]
					if type(expt[0]) is list:
						if expt[0][1]=='list':
							if same_type(expt[0][0],nk):
								tree_branch[-1].append(nch)
							else:
								tree_branch.append([])
								expt = expt[1:]
								continue
						elif expt[0][1]=='option':
							assert tree_branch[-1]==[]
							if same_type(expt[0][0],nk) or nk == 'NULL':
								tree_branch[-1].append(nch)
								tree_branch.append([])
								expt = expt[1:]
							else:
								tree_branch.append([])
								expt = expt[1:]
								continue
						else:
							assert Failure
					else:
						et = expt[0]
						assert same_type(et,nk)
						assert tree_branch[-1]==[]
						tree_branch[-1].append(nch)
						tree_branch.append([])
						expt = expt[1:]
					i += 1
				
				if tree_branch[-1]==[]:
					tree_branch = tree_branch[:-1]
				tree_branch += [[] for _ in expt]
				assert all(map(lambda x: x[1] in ['list','option'],expt))
				
				assert len(tree_branch)==len(bexpt)
				sl.astdata = sl.gen_astdata(tree_branch,bexpt)
				return sl.astdata
			except AssertionError:
				print('mousiran')
				raise AssertionError
				print('failure')
				print(sl.kind)
				print(bexpt,cks)
				sl.debug()
	
	def gen_asdata(sl,tree_branch,expt):
		res = []
		for k,v in zip(expt,tree_branch):
			tidx = c_cfg.nodetype2idx(k)
			if type(k) is str:
				if k in nont_reduce.keys():
					assert len(v)==1
					idx = node_reduce[k].index(v[0])
					res.append((tidx,idx,[v[0].astdata]))
				elif k == 'type':
					res.append(c_cfg.tystr2data(v[0].leafdata))
				else:
					idx = c_cfg.leafdata2idx(tidx,v[0].leafdata)
					res.append((tidx,idx,[]))
			else:
				if k[1]=='option':
					if v == []:
						res.append((tidx,0,[]))
					else:
						res.append((tidx,1,[v[0].astdata]))
				else:
					assert k[1]=='list'
					nre = (tidx,0,[])
					for cv in v[::-1]:
						nre = res.append((tidx,1,[nre,cv.astdata]))
					res.append(nre)
		
		return (c_cfg.nodetype2idx(sl.kind),0,res)
	
	def nullast():
		return AST(None,isnull=True)

	def load_astdata(sl,data,terminal_ids):
		idx,mvnum,cs = data
		assert mvnum == 0
		sl.invalid = False
		sl.kind = c_cfg.idx2nodetype(idx)
		assert len(cs)==len(c_cfg.cfg[sl.kind])
		tcs = []
		for ch,ty in zip(cs,c_cfg.cfg[sl.kind]):
			cty,cidx,ccs = ch
			assert c_cfg.idx2nodetype(cty)==ty
			if type(ty) is str:
				assert len(ccs)==1
				if ty in nont_reduce.keys():
					tty = nont_reduce[ty][cidx]
					addast = sl.nullast().load_astdata(ccs[0],terminal_ids)
					assert addast.kind == tty
					tcs.append(addast)
				elif ty=='type':
					tcs.append(AST(('type',c_cfg.data2tystr(ch))))
				else:
					assert len(ccs)==0
					tcs.append(AST((ty,terminal_ids[ty][idx])))
			else:
				if ty[1]=='option':
					if cid==0:
						assert len(ccs)==0
						tcs.append(AST(('NULL','')))
					else:
						assert len(ccs)==1
						tty = nont_reduce[ty[0]][ccs[0][1]]
						addast = sl.nullast().load_astdata(ccs[0],terminal_ids)
						assert addast.kind == tty
						tcs.append(addast)
				elif ty[1]=='list':
					tty = nont_reduce[ty[0]][cidx]
					while cidx != 0:
						assert len(ccs)==2
						tty = nont_reduce[ty[0]][ccs[0][1]]
						addast = sl.nullast().load_astdata(ccs[0],terminal_ids)
						assert addast.kind == tty
						tcs.append(addast)
						cty,cidx,ccs = ccs[1]
					assert len(ccs)==0
				else:
					assert False
		sl.children = tcs
		return sl
	
	def subexpr_line_list(sl):
		res = []
		for c in sl.children:
			res += c.subexpr_line_list()
		
		if not sl in [CursorKind.COMPOUND_STMT]:
			res.append((sl.top,sl.bottom),(c_cfg.rootidx,0,sl.asdata))
		return res


def nullast():
	return AST(None,isnull=True)
