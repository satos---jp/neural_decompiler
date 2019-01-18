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
		"""
		t,b = sl.top,sl.bottom
		print('source range:',t,b)
		with open(sl.filename) as fp:
			src = fp.readlines()[t-1:b]
		print(' '.join(src))
		"""
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
		sl.astdata = None
		sl.size = 0
		if isnull:
			sl.kind = None
			sl.isnullast = True
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
			if sl.kind == CursorKind.STRUCT_DECL:
				nana = 'structname'
			else:
				nana = 'unionname'
			tcs = [AST((nana,na)),] + tcs
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
			tcs = [AST(('enumname',na))] + tcs
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
	
	
	
	def get_alignmented_children(sl):
		expt = c_cfg.cfg[sl.kind]
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
		assert all(map(lambda x: x[1] in ['list','option'],expt))
		
		tree_branch += [[] for _ in range(min(len(expt),len(bexpt)-len(tree_branch)))]
		#print(tree_branch,bexpt)
		assert len(tree_branch)==len(bexpt)
		
		return tree_branch,bexpt
		
	
	def get_astdata(sl):
		if sl.astdata is not None:
			return sl.astdata
		#print(sl.kind)
		if type(sl.kind) is str:
			sl.size = 1
			tidx = c_cfg.nodetype2idx(sl.kind)
			if sl.kind == 'type':
				tasd,nsize = c_cfg.tystr2data(sl.leafdata)
				#print(tcast)
				sl.isleaf = False
				sl.kind = c_cfg.idx2nodetype(tasd[0])
				sl.astdata = tasd
				#print('retkind',sl.astdata.kind)
				sl.size = nsize
			elif sl.kind in c_cfg.nont_reduce.keys():
				assert False
				"""
				elif sl.kind in c_cfg.nont_reduce.keys():
					code.interact(local={'k':k,'v':v})
					assert len(sl.children)==0:
					mc = sl.children[0]
					idx = c_cfg.nont_reduce[k].index(sl.kind)
					sl.astdata = (tidx,idx,[sl.children[0].astdata])
				"""
			else:
				idx = c_cfg.leafdata2idx(sl.kind,sl.leafdata)
				sl.astdata = (tidx,idx,[])
			return sl.astdata
		else:
			tree_branch,expt = sl.get_alignmented_children()
			databody = []
			for k,v in zip(expt,tree_branch):
				databody.append(sl.gen_astdata(k,v))
			#print('databody',databody,expt,tree_branch)
			sl.astdata = (c_cfg.nodetype2idx(sl.kind),0,databody)
			sl.size = 1
			for c in sl.children:
				#print(c.kind)
				sl.size += c.size
			return sl.astdata
	
	def getmid(sl,k,v):
		if k in c_cfg.nont_reduce.keys():
			#v.get_astdata()
			#print(k,v,v.kind)
			mi = c_cfg.nont_reduce[k].index(v.kind)
			d = [v.get_astdata()]
		else:
			#print(k,v,v.leafdata)
			mi = c_cfg.leafdata2idx(k,v.leafdata)
			d = []
		return mi,d
	
	def gen_astdata(sl,k,v):
		tidx = c_cfg.nodetype2idx(k)
		if type(k) is str:
			assert len(v)==1
			v[0].get_astdata()
			#print(k,v[0].kind)
			if k=='type' and v[0].kind == 'type':
				code.interact(local={'k':k,'v':v})
			#if k=='type':
			#	res = (tidx,,v[0].get_astdata())
			#else:
			mi,d = sl.getmid(k,v[0])
			res = (tidx,mi,d)
		else:
			k0idx = c_cfg.nodetype2idx(k[0])
			if k[1]=='option':
				if v == [] or v[0].kind=='NULL':
					res = (tidx,0,[])
				else:
					#print(k[0],v[0],v[0].kind)
					mi,d = sl.getmid(k[0],v[0])
					res = (tidx,1,[(k0idx,mi,d)])
			elif k[1]=='list':
				nre = (tidx,0,[])
				for cv in v[::-1]:
					#print(k[0],cv.kind)
					#assert cv.kind in c_cfg.nont_reduce[k[0]]
					#if type(cv.kind) is str:
					#	nre = (tidx,1,[cv.get_astdata(),nre])
					#else:
					#code.interact(local={'cv':cv,'k':k})
					mi,d = sl.getmid(k[0],cv)
					nre = (tidx,1,[(k0idx,mi,d),nre])
				res = nre
			else:
				assert False
		#print(k,v,res)
		return res
	
	def nullast(sl):
		return AST(None,isnull=True)

	def load_astdata_unit(sl,ty,ch):
		cty,cidx,ccs = ch
		if cidx<0:
			return AST(('TRANCATE','TRANCATE'))
		
		#print(ty,ccs)
		assert type(ty) is str
		#if ty != 'type':
		#assert len(ccs)==1
		if ty=='type':
			res = AST(('type',c_cfg.data2tystr(ccs[0])))
		elif ty in c_cfg.nont_reduce.keys():
			tty = c_cfg.nont_reduce[ty][cidx]
			addast = sl.nullast().load_astdata(ccs[0])
			#assert addast.kind == tty
			res = addast
		else:
			assert len(ccs)==0
			#res = AST((ty,c_cfg.terminal_ids[ty][idx]))
			res = AST((ty,c_cfg.idx2leafdata(ty,cidx)))
		return res
	
	def load_astdata(sl,data):
		#print(data)
		idx,mvnum,cs = data
		if mvnum<0:
			return AST(('TRANCATE','TRANCATE'))
		
		assert mvnum == 0
		sl.invalid = False
		sl.kind = c_cfg.idx2nodetype(idx)
		#print('load data',sl.kind)
		#c_cfg.data_miyasui(data)
		
		if type(sl.kind) is str:
			return sl.load_astdata_unit(sl.kind,cs[0])
		"""
		if not type(sl.kind) is list:
			assert len(cs)==len(c_cfg.cfg[sl.kind])
		else:
			if sl.kind[1]=='list':
				pass
			elif sl.kind[1]=='option':
				pass
			else:
				assert False
		"""
		tcs = []
		aligned_tcs = []
		for ch,ty in zip(cs,c_cfg.cfg[sl.kind]):
			#print('chty',ty,ch)
			if type(ty) is str:
				d = sl.load_astdata_unit(ty,ch)
				tcs.append(d)
				aligned_tcs.append(d)
			else:
				cty,cidx,ccs = ch
				if cidx < 0:
					tcs.append(AST(('TRANCATE','TRANCATE')))
					aligned_tcs.append(AST(('TRANCATE','TRANCATE')))
					continue
				if ty[1]=='option':
					if cidx==0:
						assert len(ccs)==0
						tcs.append(AST(('NULL','')))
						aligned_tcs.append(AST(('NULL','')))
					else:
						assert len(ccs)==1
						v = sl.load_astdata_unit(ty[0],ccs[0])
						tcs.append(v)
						aligned_tcs.append(v)
				elif ty[1]=='list':
					av = []
					while cidx > 0:
						assert len(ccs)==2
						d = sl.load_astdata_unit(ty[0],ccs[0])
						tcs.append(d)
						av.append(d)
						cty,cidx,ccs = ccs[1]
					if cidx<0:
						d = AST(('TRANCATE','TRANCATE'))
						tcs.append(d)
						av.append(d)
					else:
						assert len(ccs)==0
					aligned_tcs.append(av)
				else:
					assert False
		sl.children = tcs
		sl.aligned_children = aligned_tcs
		#print('check_aligned',aligned_tcs,c_cfg.cfg[sl.kind])
		return sl
	
	def subexpr_line_list(sl):
		#sl.debug()
		#print(sl.kind,type(sl.kind))
		if (not type(sl.kind) is str) and sl.kind.__module__ == 'pycparser.c_ast':
			return []
		res = []
		for c in sl.children:
			res += c.subexpr_line_list()
		
		if (not type(sl.kind) is str) and (sl.kind.__module__ == 'clang.cindex') and (not sl.kind in [CursorKind.COMPOUND_STMT]):
			#print(sl.kind,sl.kind.__module__)
			ki = c_cfg.nodetype2idx(sl.kind)
			idx = c_cfg.trans_array[c_cfg.rootidx].index([ki])
			res.append(((sl.top,sl.bottom),sl.size,(c_cfg.rootidx,idx,[sl.astdata])))
		return res
	
	
	def before_show(sl):
		if sl.kind != CursorKind.FOR_STMT:
			sl.children = list(filter(lambda x: x.kind != 'NULL',sl.children))
		
		for c in sl.children:
			c.before_show()

def nullast():
	return AST(None,isnull=True)
	
	
def load_astdata(data):
	idx,mvnum,cs = data
	assert idx==c_cfg.rootidx and len(cs)==1
	#print(cs[0][0])
	return nullast().load_astdata(cs[0])


