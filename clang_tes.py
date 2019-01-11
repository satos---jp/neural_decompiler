#! /usr/bin/env /home/satos/sotsuron/neural_decompiler/python_fix_seed.sh

from clang.cindex import Config,Index,CursorKind
Config.set_library_path("/usr/lib/llvm-8/lib/")

import code,os

kinds = set([])
rules = set([])

cnt = 0

def traverse(x,d,fnsrc):
	global kinds,rules,cnt
	def mp(a):
		print('\t' * d + str(a))
	#mp(x.kind)
	#mp(x.extent)
	#mp(x.spelling)
	#mp(x.displayname)
	#mp(x.get_usr())
	kinds |= set([x.kind])
	cs = list(x.get_children())
	#rule = (str(x.kind),tuple(map(lambda a: str(a.kind),cs)))
	rule = (str(x.kind),len(cs))
	rules |= set([rule])
	
	if x.kind == CursorKind.CONST_ATTR:# and len(cs)>=2:
		tfn,src = fnsrc
		st,ed = x.extent.start.line,x.extent.end.line
		nsrcs = src[st-1:ed]
		tcs = tuple(map(lambda a: a.kind,cs))
		print(tfn,st,ed)
		print(x.kind,tcs)
		print(''.join(nsrcs))
		print(''.join(map(lambda a: a[:-1],nsrcs)))
		print("----------around--------------")
		print(''.join(map(lambda a: a[:-1],src[max(st-1-10,0):ed+10])))
		code.interact(local={'x':x,'cs':cs})
		cnt += 1
		if cnt > 100:
			exit()
	
	if len(cs)>0:
		for c in cs:
			traverse(c,d+1,fnsrc)
	else:
		pass
		#exit()


srcs = list(map(lambda x: x[:-5],filter(lambda x: x[-5:]=='.objd',os.listdir('./build'))))

for fn in srcs:
	# due to _Static_assert
	if fn in [
		'de1e3cd1eb8fc3537c4fac3030894adbb3e763d2',
		'aa823218b622eee46297e7c22a13d36a784f1d7a',
		'120b734783157860b3bbca0164704e03c6e1c5bb',
		'7f7e8e1643296d31fce7d6fa2144f70ea3c4bc5b',
	]:
		continue
	
	tfn = 'build/' + fn + '.tokenized.c'
	with open(tfn) as fp:
		src = fp.readlines()
	index = Index.create()
	ast = index.parse(tfn)

	try:
		traverse(ast.cursor,0,(tfn,src))
	except ValueError as x:
		print('error',x,'at',tfn)
		exit()


exit()

"""
[CursorKind.ADDR_LABEL_EXPR, CursorKind.FOR_STMT, CursorKind.StmtExpr, CursorKind.GOTO_STMT, CursorKind.INDIRECT_GOTO_STMT, CursorKind.FUNCTION_DECL, CursorKind.CONTINUE_STMT, CursorKind.TYPE_REF, CursorKind.BREAK_STMT, CursorKind.VAR_DECL, CursorKind.VISIBILITY_ATTR, CursorKind.RETURN_STMT, CursorKind.ASM_STMT, CursorKind.UNEXPOSED_DECL, CursorKind.CSTYLE_CAST_EXPR, CursorKind.STRUCT_DECL, CursorKind.MEMBER_REF, CursorKind.LABEL_REF, CursorKind.UNION_DECL, CursorKind.COMPOUND_LITERAL_EXPR, CursorKind.CXX_UNARY_EXPR, CursorKind.ENUM_DECL, CursorKind.UNEXPOSED_EXPR, CursorKind.PARM_DECL, CursorKind.TYPEDEF_DECL, CursorKind.FIELD_DECL, CursorKind.DECL_REF_EXPR, CursorKind.MEMBER_REF_EXPR, CursorKind.CALL_EXPR, CursorKind.NULL_STMT, CursorKind.INTEGER_LITERAL, CursorKind.DECL_STMT, CursorKind.TRANSLATION_UNIT, CursorKind.ENUM_CONSTANT_DECL, CursorKind.FLOATING_LITERAL, CursorKind.UNEXPOSED_ATTR, CursorKind.IMAGINARY_LITERAL, CursorKind.STRING_LITERAL, CursorKind.CHARACTER_LITERAL, CursorKind.PAREN_EXPR, CursorKind.LABEL_STMT, CursorKind.UNARY_OPERATOR, CursorKind.COMPOUND_STMT, CursorKind.ARRAY_SUBSCRIPT_EXPR, CursorKind.CASE_STMT, CursorKind.BINARY_OPERATOR, CursorKind.DEFAULT_STMT, CursorKind.COMPOUND_ASSIGNMENT_OPERATOR, CursorKind.IF_STMT, CursorKind.ASM_LABEL_ATTR, CursorKind.CONDITIONAL_OPERATOR, CursorKind.PACKED_ATTR, CursorKind.SWITCH_STMT, CursorKind.PURE_ATTR, CursorKind.WHILE_STMT, CursorKind.CONST_ATTR, CursorKind.DO_STMT, CursorKind.INIT_LIST_EXPR]
58
"""


def dump_set(s):
	s = list(s)
	for a in s:
		print(a)
	print(len(s))

dump_set(kinds)
dump_set(list(sorted(rules)))
