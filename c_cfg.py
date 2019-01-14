from clang.cindex import CursorKind


node_id_data = {

}

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
		CursorKind.TYPEDEF_DECL,
		CursorKind.UNION_DECL,
		CursorKind.VAR_DECL,
		CursorKind.FUNCTION_DECL,
	],
}


reduced_type = dict(sum(map(lambda ds: list(map(lambda v: (v,ds[0]),ds[1])), nont_reduce.items()),[]))
#def get_node_type

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
	CursorKind.CXX_UNARY_EXPR: ['sizeof_kasou_var'],  # only sizeof like things. TODO(satos) restore from 8
	CursorKind.DECL_REF_EXPR: ['name'],
	CursorKind.DECL_STMT: [['decl','list']],
	CursorKind.DEFAULT_STMT: ['stmt'],
	CursorKind.DO_STMT: ['stmt','expr'],
	CursorKind.ENUM_CONSTANT_DECL: ['name',['expr','option']],
	CursorKind.ENUM_DECL: ['name',['decl','list']],
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
	CursorKind.STRUCT_DECL: ['name',['decl','list']],
	CursorKind.SWITCH_STMT: ['expr','stmt'],
	CursorKind.StmtExpr: None, # return ({int x = 3;4;});   GNU extension?, skip.
	CursorKind.TRANSLATION_UNIT: [['decl','list']],
	CursorKind.TYPEDEF_DECL: None, # remove typedef information (normalize)
	CursorKind.TYPE_REF: ['var'],
	CursorKind.UNARY_OPERATOR: ['unop','expr'], #what is difference with CursorKind.CXX_UNARY_EXPR
	CursorKind.UNEXPOSED_ATTR: None, # __attribute__((__nothrow__))   skip. (or remove?)
	CursorKind.UNEXPOSED_EXPR: ['expr'], # Implicit cast (len==1) or initialize struct like .d=3 (len==2) or other insane expressions
	CursorKind.UNION_DECL: ['name',['decl','list']],
	CursorKind.VAR_DECL: ['type','name',['expr','option']], # ayasii
	CursorKind.VISIBILITY_ATTR: None, # __attribute__((visibility("hidden"))). skip.
	CursorKind.WHILE_STMT: ['expr','stmt'],
	CursorKind.UNEXPOSED_DECL: None, #mousiran
}

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
			assert False
	
	return f(td)
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
		raise Oteage
		print('function_decl_func_fisrt_miss')
		print(cs)
		exit()
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
	assert len(cs)<=1
	if len(cs)==0:
		body = ';'
	else:
		body = cs[0]
	
	try:
		return td[0] + na + '(' + ','.join(map(lambda ab: type_embedding(*ab),arg_names)) + ')' + body
	except TypeError:
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
	if ';' in s[:-1]:
		raise Oteage
	return 'for(' + cs[0] + ('' if len(s)>0 and s[-1]==';' else ';') + '{1};{2}){3}'.format(*cs)

cfg2str = {
	CursorKind.ARRAY_SUBSCRIPT_EXPR: '{0}[{1}]',
	CursorKind.BINARY_OPERATOR: '{0} {1} {2}', 
	CursorKind.BREAK_STMT: 'break;\n',
	CursorKind.CALL_EXPR: (lambda cs: cs[0] + '(' + ','.join(cs[1:]) + ')'),
	CursorKind.CASE_STMT: (lambda cs: 'case ' + cs[0] + ':\n' + stmtize(cs[1])),
	CursorKind.CHARACTER_LITERAL: ['literal'],
	CursorKind.COMPOUND_ASSIGNMENT_OPERATOR: '{0} {1} {2}', 
	CursorKind.COMPOUND_STMT: (lambda cs: '{\n%s}\n' % stmts2str(cs)),	
	#(lambda cs: ('%s' if len(cs)==1 else '{\n%s}\n') % stmts2str(cs)),	#(lambda cs: '{\n\t' + ''.join(cs).replace('\n','\n\t')[:-1] + '}\n'),
	CursorKind.CONDITIONAL_OPERATOR: '{0} ? {1} : {2}', 
	CursorKind.CONTINUE_STMT: 'continue;',
	CursorKind.CSTYLE_CAST_EXPR: '({0}){1}', #TODO(satos) ayasii  -> # retrive type from .type.get_canonical().spelling 
	CursorKind.CXX_UNARY_EXPR: '8', #TODO(satos) maybe later ['unary_op','expr_option'],  # only sizeof like things.
	CursorKind.DECL_REF_EXPR: '{0}',
	CursorKind.DECL_STMT: (lambda cs: stmts2str(cs)),
	CursorKind.DEFAULT_STMT: 'default:\n{0};\n',
	CursorKind.DO_STMT: 'do{{{0}}}while({1})',
	CursorKind.ENUM_CONSTANT_DECL: (lambda cs: cs[0] + ('' if len(cs)==1 else ' = ' + cs[1])),  # akirameru
	CursorKind.ENUM_DECL: (lambda cs: 'enum ' + cs[0] + '{' + ','.join(cs[1:]) + '};'),             # akirameru
	CursorKind.FIELD_DECL: (lambda cs: type_embedding(cs[0],cs[1]) + ';\n'), # [int hogehuga] in [struct{int hogehuga};]
	CursorKind.FLOATING_LITERAL: ['literal'],
	CursorKind.FOR_STMT: for_check,
	CursorKind.FUNCTION_DECL: function_decl_func,
	CursorKind.GOTO_STMT: 'goto {0};',
	CursorKind.IF_STMT: (lambda cs: 'if(' + cs[0] + ')' + stmtize(cs[1]) + ('' if len(cs)==2 else 'else ' + cs[2])),
	CursorKind.IMAGINARY_LITERAL: None, # TODO(satos) imaginary number. currently omit. 
	CursorKind.INDIRECT_GOTO_STMT: None, # TODO(satos) like goto *hoge. Is this valid in standard C?
	CursorKind.INIT_LIST_EXPR: (lambda cs: '{' + ','.join(cs) + '}'), # TODO(satos) sometimes too long, restrict?
	CursorKind.INTEGER_LITERAL: ['literal'],
	CursorKind.LABEL_REF: '{0}',
	CursorKind.LABEL_STMT: '{0}:\n{1}',
	CursorKind.MEMBER_REF: ['var'], # x in {.x=3};
	CursorKind.MEMBER_REF_EXPR: '{0}{1}{2}',
	CursorKind.NULL_STMT: ';',
	CursorKind.PAREN_EXPR: '({0})',
	CursorKind.PARM_DECL: (lambda cs: cs), # f(int,int)    argument type. 
	CursorKind.RETURN_STMT: (lambda cs: 'return;' if len(cs)==0 else 'return ' + cs[0] + ';'),
	CursorKind.STRING_LITERAL: ['literal'],
	CursorKind.STRUCT_DECL: (lambda cs: (('struct ' + cs[0] + '{' + ''.join(cs[1:]) + '}' + ';\n') if len(cs)>1 else '')),
	CursorKind.SWITCH_STMT: 'switch({0}){1}',
	CursorKind.TRANSLATION_UNIT: stmts2str,
	#CursorKind.TYPEDEF_DECL: (lambda cs: 'typedef ' + type_embedding(cs[0],cs[1])), #ayasii
	#CursorKind.TYPE_REF: ['var'],
	CursorKind.UNARY_OPERATOR: (lambda cs: ('{0} {1}' if cs[0][1] else '{1} {0}').format(cs[0][0],cs[1])), #what is difference with CursorKind.CXX_UNARY_EXPR
	CursorKind.UNEXPOSED_EXPR: (lambda cs: cs[0] if len(cs)==1 else 'INVALID'), # Implicit cast (len==1) or initialize struct like .d=3 (len==2) or other insane expressions
	CursorKind.UNION_DECL: (lambda cs: 'union ' + cs[0] + '{' + ''.join(cs[1:]) + '};\n'),
	CursorKind.VAR_DECL: (lambda cs: type_embedding(cs[0],cs[1]) + ('=' + cs[2] if len(cs)==3 else '') + ';'),
	CursorKind.WHILE_STMT: 'while({0}){1}',
}


	
	
	
	




