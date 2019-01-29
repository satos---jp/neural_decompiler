from clang.cindex import CursorKind
import pycparser
import my_c_ast

# liteeal can be expr
# expr can be stmt
nont_reduce = {
	'toplevel': [CursorKind.TRANSLATION_UNIT],
	#'type': CursorKind.TYPE_REF,
	'labelref': [CursorKind.LABEL_REF],
	'expr': [
		CursorKind.ARRAY_SUBSCRIPT_EXPR,
		CursorKind.BINARY_OPERATOR,
		CursorKind.CALL_EXPR,
		CursorKind.CHARACTER_LITERAL,
		CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,
		CursorKind.CONDITIONAL_OPERATOR,
		CursorKind.CSTYLE_CAST_EXPR,
		CursorKind.CXX_UNARY_EXPR,  # only for sizeof
		CursorKind.DECL_REF_EXPR,
		CursorKind.FLOATING_LITERAL,
		CursorKind.INIT_LIST_EXPR,
		CursorKind.INTEGER_LITERAL,
		CursorKind.MEMBER_REF_EXPR,
		CursorKind.PAREN_EXPR,
		CursorKind.STRING_LITERAL,
		CursorKind.UNARY_OPERATOR,
		CursorKind.UNEXPOSED_EXPR,
		CursorKind.STRING_LITERAL,
	],
	'stmt': [
		CursorKind.BREAK_STMT,
		CursorKind.CASE_STMT,
		CursorKind.COMPOUND_STMT,
		CursorKind.CONTINUE_STMT,
		CursorKind.DECL_STMT,
		CursorKind.DEFAULT_STMT,
		CursorKind.DO_STMT,
		CursorKind.FOR_STMT,
		CursorKind.GOTO_STMT,
		CursorKind.IF_STMT,
		CursorKind.LABEL_STMT,
		CursorKind.NULL_STMT,
		CursorKind.RETURN_STMT,
		CursorKind.SWITCH_STMT,
		CursorKind.WHILE_STMT,
		CursorKind.WHILE_STMT,
	],
	'decl': [
		CursorKind.ENUM_CONSTANT_DECL,
		CursorKind.ENUM_DECL,
		CursorKind.FIELD_DECL,
		CursorKind.PARM_DECL,
		CursorKind.STRUCT_DECL,
		#CursorKind.TYPEDEF_DECL,
		CursorKind.UNION_DECL,
		CursorKind.VAR_DECL,
		CursorKind.FUNCTION_DECL,
	],
	
	#from c_type_cfg
	'type_type': [
		pycparser.c_ast.ArrayDecl,
		pycparser.c_ast.FuncDecl,
		pycparser.c_ast.TypeDecl,
		pycparser.c_ast.PtrDecl,
	],
	'paramlist': [
		pycparser.c_ast.ParamList
	],
	'type': [
		pycparser.c_ast.Typename
	],
	'typename': [
		pycparser.c_ast.Typename
	],
	'typedecltype': [
		pycparser.c_ast.Struct,
		pycparser.c_ast.Union,
		pycparser.c_ast.Enum,
		pycparser.c_ast.IdentifierType,
	],
}

reduced_type = dict(sum(map(lambda ds: list(map(lambda v: (v,ds[0]),ds[1])), nont_reduce.items()),[]))
#def get_node_type

for v in nont_reduce['expr']:
	nont_reduce['stmt'].append(v)

#print(nont_reduce)

# None is for pass.
cfg = {
	CursorKind.ADDR_LABEL_EXPR: None,
	CursorKind.ARRAY_SUBSCRIPT_EXPR: ['expr','expr'],
	CursorKind.ASM_LABEL_ATTR: None,	
	CursorKind.ASM_STMT: None,
	CursorKind.BINARY_OPERATOR: ['expr','binop','expr'], 
	CursorKind.BREAK_STMT: [],
	CursorKind.CALL_EXPR: ['expr',['expr','list']],
	CursorKind.CASE_STMT: ['expr','stmt'],
	CursorKind.CHARACTER_LITERAL: ['charliteral'],
	CursorKind.COMPOUND_ASSIGNMENT_OPERATOR: ['expr','casop','expr'],
	CursorKind.COMPOUND_LITERAL_EXPR: None, # TODO(satos)   (struct S){.x=1}  or (int[]){1,2,3,some_int} like.
	CursorKind.COMPOUND_STMT: [['stmt','list']],
	CursorKind.CONDITIONAL_OPERATOR: ['expr','expr','expr'],
	CursorKind.CONST_ATTR: None, # __attribute__((__const__))   skip. (or remove?)
	CursorKind.CONTINUE_STMT: [],
	CursorKind.CSTYLE_CAST_EXPR: ['type','expr'], #TODO(satos) ayasii  -> # retrive type from .type.get_canonical().spelling 
	CursorKind.CXX_UNARY_EXPR: [],  # only sizeof like things. TODO(satos) restore from 8
	CursorKind.DECL_REF_EXPR: ['name'],
	CursorKind.DECL_STMT: [['decl','list']],
	CursorKind.DEFAULT_STMT: ['stmt'],
	CursorKind.DO_STMT: ['stmt','expr'],
	CursorKind.ENUM_CONSTANT_DECL: ['name',['expr','option']],
	CursorKind.ENUM_DECL: ['enumname',['decl','list']],
	CursorKind.FIELD_DECL: ['type','name',['expr','option']], # [int hogehuga] in [struct{int hogehuga};]
	CursorKind.FLOATING_LITERAL: ['floatliteral'],
	CursorKind.FOR_STMT: [['stmt','option'],['expr','option'],['expr','option'],'stmt'],
	CursorKind.FUNCTION_DECL: ['type','name',['decl','list'],['stmt','option']], # TODO(satos) ayasii
	CursorKind.GOTO_STMT: ['labelref'],
	CursorKind.IF_STMT: ['expr','stmt',['stmt','option']],
	CursorKind.IMAGINARY_LITERAL: None, # TODO(satos) imaginary number. currently omit. 
	CursorKind.INDIRECT_GOTO_STMT: None, # TODO(satos) like goto *hoge. Is this valid in standard C?
	CursorKind.INIT_LIST_EXPR: [['expr','list']], # TODO(satos) sometimes too long, restrict?
	CursorKind.INTEGER_LITERAL: ['integerliteral'],
	CursorKind.LABEL_REF: ['label'],
	CursorKind.LABEL_STMT: ['label','stmt'],
	CursorKind.MEMBER_REF: ['var'], # x in {.x=3};
	CursorKind.MEMBER_REF_EXPR: ['expr','memberrefop','name'],
	CursorKind.NULL_STMT: [],
	CursorKind.PACKED_ATTR: None, # __attribute__((packed))   skip. (or remove?)
	CursorKind.PAREN_EXPR: ['expr'],
	CursorKind.PARM_DECL: ['type','name'], # f(int,int)    argument type. 
	CursorKind.PURE_ATTR: None, # __attribute__((__pure__))    skip. (or remove?)
	CursorKind.RETURN_STMT: [['expr','option']],
	CursorKind.STRING_LITERAL: ['stringliteral'],
	CursorKind.STRUCT_DECL: ['structname',['decl','list']],
	CursorKind.SWITCH_STMT: ['expr','stmt'],
	CursorKind.StmtExpr: None, # return ({int x = 3;4;});   GNU extension?, skip.
	CursorKind.TRANSLATION_UNIT: [['decl','list']],
	CursorKind.TYPEDEF_DECL: None, # remove typedef information (normalize)
	CursorKind.TYPE_REF: ['var'],
	CursorKind.UNARY_OPERATOR: ['unop','expr'], #what is difference with CursorKind.CXX_UNARY_EXPR
	CursorKind.UNEXPOSED_ATTR: None, # __attribute__((__nothrow__))   skip. (or remove?)
	CursorKind.UNEXPOSED_EXPR: ['expr'], # Implicit cast (len==1) or initialize struct like .d=3 (len==2) or other insane expressions
	CursorKind.UNION_DECL: ['unionname',['decl','list']],
	CursorKind.VAR_DECL: ['type','name',['expr','option']], # ayasii
	CursorKind.VISIBILITY_ATTR: None, # __attribute__((visibility("hidden"))). skip.
	CursorKind.WHILE_STMT: ['expr','stmt'],
	CursorKind.UNEXPOSED_DECL: None, #mousiran
}
"""
	# from c_type_cfg
	pycparser.c_ast.Typename: [['tyspec','list'],'type'],
	
	pycparser.c_ast.ArrayDecl: [['dim_expr','option'],'type'],
	pycparser.c_ast.TypeDecl: [['tyspec','list'],'typedecltype'],
	pycparser.c_ast.PtrDecl: [['tyspec','list'],'type'],
	pycparser.c_ast.FuncDecl: [['paramlist','option'],'type'],
	pycparser.c_ast.ParamList: [['typename','list']],
	
	pycparser.c_ast.Struct: ['structname'],
	pycparser.c_ast.Union: ['unionname'],
	pycparser.c_ast.IdentifierType: [['tyspec','list']],
}
"""

cfg_fortype = {
	pycparser.c_ast.Typename: [(['tyspec','list'],'quals'),('type_type','type')],
	
	pycparser.c_ast.ArrayDecl: [(['integerliteral','option'],'dim'),('type_type','type')],
	pycparser.c_ast.TypeDecl: [(['tyspec','list'],'quals'),('typedecltype','type')],
	pycparser.c_ast.PtrDecl: [(['tyspec','list'],'quals'),('type_type','type')],
	pycparser.c_ast.FuncDecl: [(['paramlist','option'],'args'),('type_type','type')],
	pycparser.c_ast.ParamList: [(['typename','list'],'params')],
	
	pycparser.c_ast.Struct: [('structname','name')],
	pycparser.c_ast.Union: [('unionname','name')],
	pycparser.c_ast.Enum: [('enumname','name')],
	pycparser.c_ast.IdentifierType: [(['tyspec','list'],'names')],
}

for k,v in cfg_fortype.items():
	cfg[k] = list(map(lambda x: x[0],v))

"""
CursorKind.CSTYLE_CAST_EXPR
CursorKind.DO_STMT
CursorKind.FLOATING_LITERAL
CursorKind.INIT_LIST_EXPR

CursorKind.CXX_UNARY_EXPR

CursorKind.ENUM_CONSTANT_DECL
CursorKind.ENUM_DECL
CursorKind.FIELD_DECL
CursorKind.STRUCT_DECL
CursorKind.UNION_DECL
"""

def get_var_length(ty):
	return 0

def s2tree_with_parenthes(s):
	r = ""
	res = []
	attrs = '__attribute__('
	while len(s)>0:
		if s[0]=='(':
			if len(r.strip())>0:
				res.append(r)
			r = ""
			d,s = s2tree_with_parenthes(s[1:])
			res.append(d)
			assert s[0]==')'
			s = s[1:]
		elif s[0]==')':
			break
		elif s[:len(attrs)]==attrs:
			_,s = s2tree_with_parenthes(s[len(attrs):])
			assert s[0]==')'
			s = s[1:]
		else:
			r += s[0]
			s = s[1:]
	if len(r.strip())>0:
		res.append(r)
	return res,s

def strize(x):
	if type(x) is str:
		return x
	else:
		return '(' + ''.join(map(strize,x)) + ')'

def type_embedding(ty,na):
	#if 'restrict' in ty:
	ty = ty.replace('restrict','__restrict')
	#if 'struct __va_list_tag' in ty:
		#print(ty,na)
	#ty = ty.replace('struct __va_list_tag','__va_list_tag')
		#print(ty)
	#print(ty,na)
	td,s = s2tree_with_parenthes(ty)
	#print(td,s)
	assert len(s)==0
	

	
	def f(nt):
		if len(nt)==1:
			if '[' in nt[0]:
				p = nt[0].index('[')
				return nt[0][:p] + ' ' + na + nt[0][p:]
			else:
				return nt[0] + ' ' + na
		elif len(nt)==2:
			return nt[0] + ' ' + na + strize(nt[1])
		elif len(nt)==3:
			return nt[0] + '(' + f(nt[1]) + ')' + strize(nt[2])
		else:
			#print('assert_at_type_embedding')
			#print(nt,len(nt),ty)
			assert False
	
	try:
		return f(td)
	except (IndexError,AssertionError,TypeError):
		return 'INDALID_TYPE ' + na
	#except AssertError:
	#return "%s %s" % (ty,na)


def function_decl_func(cs):
	#print('cs',cs)
	ty = cs[0]
	na = cs[1]
	cs = cs[2:]
	td,s = s2tree_with_parenthes(ty)
	#print(s,td)
	try:
		assert len(s)==0 and len(td)==2 and type(td[1]) is list
	except AssertionError:
		#raise Oteage
		#print('function_decl_func_fisrt_miss')
		#print(cs)
		#exit()
		#print('invalid function decl')
		#print(cs)
		#exit()
		return 'INVALID_FUNCTION_DECL{' + ';'.join(cs) + ';}'
	arg_types_len = 0
	if len(td[1])!=0 and td[1][0] != 'void':
		arg_types_len = 1
		for d in td[1]:
			if type(d) is list:
				continue
			else:
				#print(d,d.count(','))
				arg_types_len += d.count(',')
	
	#print(arg_types_len)
	arg_names = cs[:arg_types_len]
	cs = cs[arg_types_len:]
	#assert len(cs)<=1
	if len(cs)==0:
		body = ';'
	else:
		body = cs[0]
	
	try:
		return td[0] + na + '(' + ','.join(arg_names) + ')' + body
	except TypeError:
		return 'INVALID_FUNCTION_DECL{' + ''.join(cs) + '}'
		print('nazono type embedding no yatu')
		raise AssertionError
		print(cs)
		print(arg_names)
		exit()

def stmts2str(cs):
	res = ''
	for d in cs:
		ds = d.strip()
		if len(ds)==0 or ds[-1] in [';','}']:
			res += d + '\n'
		else:
			res += d + ';\n'
	return res

def stmtize(s):
	if s.strip()[-1] in [';','}']:
		return s 
	else:
		return s + ';'


class Oteage(Exception):
	pass

def for_check(cs):
	s = cs[0].strip()
	#if ';' in s[:-1]:
	#	raise Oteage
	return 'for(' + cs[0] + ('' if len(s)>0 and s[-1]==';' else ';') + '{1};{2})'.format(*cs) + stmtize(cs[3])

cfg2str = {
	CursorKind.ARRAY_SUBSCRIPT_EXPR: '{0}[{1}]',
	CursorKind.BINARY_OPERATOR: '{0} {1} {2}', 
	CursorKind.BREAK_STMT: 'break;\n',
	CursorKind.CALL_EXPR: (lambda cs: cs[0] + '(' + ','.join(cs[1:]) + ')'),
	CursorKind.CASE_STMT: (lambda cs: 'case ' + cs[0] + ':\n' + stmtize(cs[1])),
	CursorKind.CHARACTER_LITERAL: '{0}',
	CursorKind.COMPOUND_ASSIGNMENT_OPERATOR: '{0} {1} {2}', 
	CursorKind.COMPOUND_STMT: (lambda cs: '{\n%s}\n' % stmts2str(cs)),	
	#(lambda cs: ('%s' if len(cs)==1 else '{\n%s}\n') % stmts2str(cs)),	#(lambda cs: '{\n\t' + ''.join(cs).replace('\n','\n\t')[:-1] + '}\n'),
	CursorKind.CONDITIONAL_OPERATOR: '{0} ? {1} : {2}', 
	CursorKind.CONTINUE_STMT: 'continue;',
	CursorKind.CSTYLE_CAST_EXPR: '({0}){1}', #TODO(satos) ayasii  -> # retrive type from .type.get_canonical().spelling 
	CursorKind.CXX_UNARY_EXPR: '8', #TODO(satos) maybe later ['unary_op','expr_option'],  # only sizeof like things.
	CursorKind.DECL_REF_EXPR: '{0}',
	CursorKind.DECL_STMT: (lambda cs: stmts2str(cs)),
	CursorKind.DEFAULT_STMT: (lambda cs: 'default:\n' + stmtize(cs[0])),
	CursorKind.DO_STMT: 'do{{{0}}}while({1})',
	CursorKind.ENUM_CONSTANT_DECL: (lambda cs: cs[0] + ('' if len(cs)==1 else ' = ' + cs[1])),  # akirameru
	CursorKind.ENUM_DECL: (lambda cs: 'enum ' + cs[0] + '{' + ','.join(cs[1:]) + '};'),             # akirameru
	CursorKind.FIELD_DECL: (lambda cs: type_embedding(cs[0],cs[1]) + ';\n'), # [int hogehuga] in [struct{int hogehuga};]
	CursorKind.FLOATING_LITERAL: '{0}',
	CursorKind.FOR_STMT: for_check,
	CursorKind.FUNCTION_DECL: function_decl_func,
	CursorKind.GOTO_STMT: 'goto {0};',
	CursorKind.IF_STMT: (lambda cs: 'if(' + cs[0] + ')' + stmtize(cs[1]) + ('' if len(cs)==2 else 'else ' + stmtize(cs[2]))),
	#CursorKind.IMAGINARY_LITERAL: None, # TODO(satos) imaginary number. currently omit. 
	#CursorKind.INDIRECT_GOTO_STMT: None, # TODO(satos) like goto *hoge. Is this valid in standard C?
	CursorKind.INIT_LIST_EXPR: (lambda cs: '{' + ','.join(cs) + '}'), # TODO(satos) sometimes too long, restrict?
	CursorKind.INTEGER_LITERAL: '{0}',
	CursorKind.LABEL_REF: '{0}',
	CursorKind.LABEL_STMT: (lambda cs: cs[0] + ':\n' + stmtize(cs[1])),
	#CursorKind.MEMBER_REF: ['var'], # x in {.x=3};
	CursorKind.MEMBER_REF_EXPR: '{0}{1}{2}',
	CursorKind.NULL_STMT: ';',
	CursorKind.PAREN_EXPR: '({0})',
	CursorKind.PARM_DECL: (lambda cs: type_embedding(*cs)), # f(int,int)    argument type. 
	CursorKind.RETURN_STMT: (lambda cs: 'return;' if len(cs)==0 else 'return ' + cs[0] + ';'),
	CursorKind.STRING_LITERAL: '{0}',
	CursorKind.STRUCT_DECL: (lambda cs: (('struct ' + cs[0] + '{' + ''.join(cs[1:]) + '}' + ';\n') if len(cs)>1 else '')),
	CursorKind.SWITCH_STMT: (lambda cs: 'switch(' + cs[0] + ')' + stmtize(cs[1])),
	CursorKind.TRANSLATION_UNIT: stmts2str,
	#CursorKind.TYPEDEF_DECL: (lambda cs: 'typedef ' + type_embedding(cs[0],cs[1])), #ayasii
	#CursorKind.TYPE_REF: ['var'],
	CursorKind.UNARY_OPERATOR: (lambda cs: ('{0} {1}' if cs[0][1] else '{1} {0}').format(cs[0][0],cs[1])), #what is difference with CursorKind.CXX_UNARY_EXPR
	CursorKind.UNEXPOSED_EXPR: (lambda cs: cs[0] if len(cs)==1 else 'INVALID'), # Implicit cast (len==1) or initialize struct like .d=3 (len==2) or other insane expressions
	CursorKind.UNION_DECL: (lambda cs: 'union ' + cs[0] + '{' + ''.join(cs[1:]) + '};\n'),
	CursorKind.VAR_DECL: (lambda cs: type_embedding(cs[0],cs[1]) + ('=' + cs[2] if len(cs)==3 else '') + ';'),
	CursorKind.WHILE_STMT: (lambda cs: 'while(' + cs[0] + ')' + stmtize(cs[1])),
}


"""
leaftypes = [
	'binop', 'stringliteral', 'integerliteral', 'unop', 'label', 
	'charliteral', 'casop', 'memberrefop', 'floatliteral',
	'tyspec',
]
"""

shortleavs = {
	'binop': ['=', '&', '==', '>', '||', '|', '-', '+', '/', '<<', '>>', '!=', '>=', '<=', '&&', ',', '%', '^', '<', '*'], 
	'unop': [
		('--', False), ('~', True), ('++', False), ('++', True), ('--', True), 
		('-', True), ('+', True), ('&', True), ('!', True), ('__extension__', True), 
		('__real__', True), ('__imag__', True), ('*', True)
	],
	'casop': ['*=', '|=', '&=', '>>=', '<<=', '-=', '^=', '%=', '/=', '+='], 
	'memberrefop': ['.', '->'], 
	'NULL': [''],
	'tyspec': [
		'double', 'float',  'int', 'long', 'const', 'static', 'register',
		'signed', 'unsigned',  'restrict',  'extern',  'short', 'void', 'volatile', 'char','inline',
		'_Bool','__int128','_Complex',
	],
}

alphaleavs = {
	'name': [],
	'structname': [],
	'unionname': [],
	'enumname': [],
	'label': [],
}

literalleaves = ['integerliteral','stringliteral','floatliteral','charliteral']

alltypes = (
	[['expr','option'],['stmt','option'],['decl','list'],['expr','list'],['stmt','list']] + 
	[['tyspec','list'],['paramlist','option'],['integerliteral','option'],['typename','list']] + 
	literalleaves +
	list(shortleavs.keys()) + list(alphaleavs.keys()) + list(nont_reduce.keys()) + 
	list(cfg2str.keys()) + list(cfg_fortype.keys()) 
)

def nodetype2idx(ty):
	#print(alltypes)
	return alltypes.index(ty)

def idx2nodetype(idx):
	return alltypes[idx]


#print(list(filter(lambda a: a[1] is None,[(k,cfg[k]) for k in cfg2str.keys()])))


max_int_zantei = 256
def transf(d):
	if type(d) is str:
		if d=='integerliteral':
			rv = [[] for _ in range(max_int_zantei+1)]
		elif d in ['stringliteral','floatliteral','charliteral']:
			rv = [[]]
	elif type(d) is list:
		if d[1]=='option':
			rv = [[],[nodetype2idx(d[0])]]
		elif d[1]=='list':
			rv = [[],[nodetype2idx(d[0]),nodetype2idx(d)]]
	
	return (d,rv)

alphaconv_size = 60
trans_array = (
	[transf(k) for k in alltypes[:13]] + 
	[(k,[[] for _ in v]) for k,v in shortleavs.items()] + 
	[(k,[[] for _ in range(alphaconv_size)]) for k,_ in alphaleavs.items()] + # 
	[(k,[[nodetype2idx(d)] for d in v]) for k,v in nont_reduce.items()] + 
	[(k,[[nodetype2idx(d) for d in cfg[k]]]) for k in cfg2str.keys()] + 
	[(k,[[nodetype2idx(d) for d in cfg[k]]]) for k in cfg_fortype.keys()] + 
	[(None,[[i] for i in filter(lambda i: ('__module__' in dir(idx2nodetype(i)) and idx2nodetype(i).__module__=='clang.cindex'),range(len(alltypes)))])]
)

assert len(trans_array)==len(alltypes)+1 and all(map(lambda ab: ab[0][0]==ab[1],zip(trans_array[:-1],alltypes)))

#print(trans_array)

trans_array = list(map(lambda x: x[1],trans_array))
#print(trans_array)

#print(list(enumerate(alltypes)))

rootidx = len(alltypes)
#alltypes.append('all_root')
#c_cfg.cfg['all_root'] = list(cfg2str.keys())

typedata_cache = {}

def init_idxnize():
	for k in alphaleavs.keys():
		alphaleavs[k] = []
	typedata_cache = {}

def leafdata2idx(ty,v):
	global max_int_zantei
	#print(ty,v)
	if ty in shortleavs.keys():
		#if ty == 'tyspec':
		#	print(v)
		return shortleavs[ty].index(v)
	elif ty in alphaleavs.keys():
		if v in alphaleavs[ty]:
			return alphaleavs[ty].index(v)
		else:
			ls = len(alphaleavs[ty])
			alphaleavs[ty].append(v)
			return ls
	elif ty in ['stringliteral','floatliteral','charliteral']:
		return 0
	elif ty == 'integerliteral':
		kn = max_int_zantei
		#print(v)
		if type(v) is str:
			v = eval(v)
		#print(v,type(v))
		if v < 0:
			raise Oteage
		assert v>=0
		if v<kn:
			return v
		else:
			return kn
	else:
		assert False



def idx2leafdata(ty,idx):
	#print(ty,v)
	if ty in shortleavs.keys():
		#if ty == 'tyspec':
		#	print(v)
		return shortleavs[ty][idx]
	elif ty in alphaleavs.keys():
		return "__ALPHA_%s_%a__" % (ty,idx)
		"""
		if v in alphaleavs[ty]:
			alphaleavs[ty][idx]
		else:
			ls = len(alphaleavs[ty])
			alphaleavs[ty].append(v)
			return ls
		"""
	elif ty in ['stringliteral','floatliteral','charliteral']:
		return '__DUMMY_%s__' % ty
	elif ty == 'integerliteral':
		return str(idx)
	else:
		assert False





import random

def alphaconv(data):
	table = {nodetype2idx(k): {} for k in alphaleavs.keys()}
	def rec(nd):
		ty,act,cs = nd
		if ty in table.keys():
			assert len(cs)==0
			if not act in table[ty].keys():
				while True:
					r = random.randint(0,alphaconv_size-1)
					if r in table[ty].values():
						continue
					table[ty][act] = r
					break
			act = table[ty][act]
		
		return (ty,act,list(map(rec,cs)))
	
	return rec(data)

from pycparser import c_parser,c_generator,plyparser
import code


parser = c_parser.CParser()
generator = c_generator.CGenerator()

def data2tystr(data):
	#print('data2tystr')
	#data_miyasui(data)
	def treenize(ast):
		ty = ast.kind
		#print(ast,ty)
		if type(ty) is str:
			if ast.kind == 'NULL':
				return None
			elif ast.kind == 'integerliteral':
				return pycparser.c_ast.Constant(type='int',value=ast.leafdata)
			else:
				return ast.leafdata
		args = {}
		for k in ty.__init__.__code__.co_varnames:
			if k=='self':
				continue
			args[k] = None
		
		cts = cfg_fortype[ty]
		acs = ast.aligned_children
		#print(acs,cts)
		assert len(acs)==len(cts)
		for (nkata,kna),v in zip(cts,acs):
			if type(nkata) is list and nkata[1]=='list' and type(v) is list:
				args[kna] = list(map(treenize,v))
			else:
				args[kna] = treenize(v)
		
		#print(ty,args)
		return ty(**args)
	
	ast = my_c_ast.nullast().load_astdata(data)	
	tree = treenize(ast)
	#print(tree)
	try:
		restr = generator.visit(tree)
	except AttributeError as e:
		#print(e)
		#code.interact(local={'tree':tree})
		return 'TRUNCATED_TYPE'
	#print('restr ::',restr)
	return restr

def tystr2data(ty):
	#print(ty)
	cache = False
	if len(ty)<7: #'int [0]'
		cache = True
	
	if ty in typedata_cache.keys():
		return typedata_cache[ty]
	
	try:
		ki = parser.parse('int _ = (' + ty + ')0;', filename='<stdin>')
	except plyparser.ParseError:
		raise Oteage
	
	ki = ki.ext[0].init.to_type
	#print(ki)
	
	def rec_conv(node,nast,ks):
		if type(node) is str:
			nast.isleaf = True
			nast.leafdata = node
			nast.kind = ks
			return nast
		#print(node)
		#print(c_type_cfg.cfg)
		#if type(node) is pycparser.c_ast.Enum:
		#	code.interact(local={'node':node})
		
		assert node is not None
		#if node is None:
		#	return my_c_ast.AST(('NULL',''))
		cs = cfg_fortype[type(node)]
		nast.kind = type(node)
		tcs = []
		#print(cs)
		for ks,k in cs:
			if type(ks) is not str:
				ks = ks[0]
			kv = node.__getattribute__(k)
			if kv is None:
				tcs.append(my_c_ast.AST(('NULL','')))
				continue
			if type(node) is pycparser.c_ast.ArrayDecl and k=='dim':
				#code.interact(local={'kv': kv,'node':node})
				# aki
				if type(kv) is pycparser.c_ast.Constant and kv.type == 'int':
					dn = eval(kv.value)
				else:
					dn = 0
				cast = my_c_ast.nullast()
				cast.kind = 'integerliteral'
				cast.isleaf = True
				cast.leafdata = dn
				tcs.append(cast)
				continue
				
			if type(kv) is list:
				for rkv in kv:
					tcs.append(rec_conv(rkv,my_c_ast.nullast(),ks))
			else:
				tcs.append(rec_conv(kv,my_c_ast.nullast(),ks))
		nast.children = tcs
		#print('rec',nast.kind)
		return nast
	
	ast = rec_conv(ki,my_c_ast.nullast(),'MITEI')
	res = ast.get_astdata(),ast.size
	if cache:
		typedata_cache[ty] = res
	#print('tystr2data')
	#data_miyasui(res[0])
	return res




def data_miyasui(data,d=0):
	ty,act,cs = data
	print('  ' * d,idx2nodetype(ty),act)
	for c in cs:
		data_miyasui(c,d+1)

def validate_astdata(data,tys):
	ty,act,cs = data
	#code.interact(local={'data':data})
	try:
		assert idx2nodetype(ty)==tys
		#print('firch')
		if type(tys) is list:
			if tys[1]=='option':
				if act==0:
					assert len(cs)==0
				else:
					assert len(cs)==1
					validate_astdata(cs[0],tys[0])
			elif tys[1]=='list':
				if act==0:
					assert len(cs)==0
				else:
					assert len(cs)==2
					validate_astdata(cs[0],tys[0])
					validate_astdata(cs[1],tys)
			else:
				assert False
		elif tys in nont_reduce.keys():
			assert len(cs)==1
			ta = nont_reduce[tys][act]
			validate_astdata(cs[0],ta)
		elif tys in cfg2str.keys() or tys in cfg_fortype.keys():
			assert act==0
			assert len(cfg[tys])==len(cs)
			#print('forl',data)
			for a,b in zip(cs,cfg[tys]):
				validate_astdata(a,b)
		elif tys in shortleavs.keys():
			assert len(cs)==0
			assert 0 <= act and act < len(shortleavs[tys])
		elif tys in alphaleavs.keys():
			assert len(cs)==0
			pass
		elif tys in literalleaves:
			pass
		else:
			assert False
	except(AssertionError):#,TypeError):
		print('assert failure',ty,idx2nodetype(ty),tys,act,len(cs))
		env = {'data':data,'tys': tys}
		print(env)
		code.interact(local=env)
		exit()







