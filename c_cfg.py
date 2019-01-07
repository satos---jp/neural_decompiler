from clang.cindex import CursorKind


# liteeal can be expr
# expr can be stmt
nont_reduce = [
	('toplevel',[
		CursorKind.TRANSLATION_UNIT
	]),
	('type',[
		CursorKind.TYPE_REF,
	]),
	('label',[
		CursorKind.LABEL_REF,
	]),
	('expr',[
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
	]),
	('stmt',[
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
	]),
	('decl',[
		CursorKind.ENUM_CONSTANT_DECL,
		CursorKind.ENUM_DECL,
		CursorKind.FIELD_DECL,
		CursorKind.PARM_DECL,
		CursorKind.STRUCT_DECL,
		CursorKind.TYPEDEF_DECL,
		CursorKind.UNION_DECL,
		CursorKind.VAR_DECL,
	]),
]

# None is for pass.
cfg = [
	(CursorKind.ADDR_LABEL_EXPR,None),
	(CursorKind.ARRAY_SUBSCRIPT_EXPR,['expr','expr']),
	(CursorKind.ASM_LABEL_ATTR,None),	
	(CursorKind.ASM_STMT,None),
	(CursorKind.BINARY_OPERATOR,['expr','binary_op','expr']), 
	(CursorKind.BREAK_STMT,[]),
	(CursorKind.CALL_EXPR,['expr','expr_list']),
	(CursorKind.CASE_STMT,['expr','stmt']),
	(CursorKind.CHARACTER_LITERAL,['literal']),
	(CursorKind.COMPOUND_ASSIGNMENT_OPERATOR,['expr','compound_assign_op','expr']),
	(CursorKind.COMPOUND_LITERAL_EXPR,None), # TODO(satos)   (struct S){.x=1}  or (int[]){1,2,3,some_int} like.
	(CursorKind.COMPOUND_STMT,['stmt_list']),
	(CursorKind.CONDITIONAL_OPERATOR,['expr','expr','expr']),
	(CursorKind.CONST_ATTR,None), #TODO(satos) pending. __const__
	(CursorKind.CONTINUE_STMT,[]),
	(CursorKind.CSTYLE_CAST_EXPR,['type_list','expr']), #TODO(satos) ayasii  -> # retrive type from .type.get_canonical().spelling 
	(CursorKind.CXX_UNARY_EXPR,['unary_op','expr_option']),  # only sizeof like things.
	(CursorKind.DECL_REF_EXPR,['var']),
	(CursorKind.DECL_STMT,['decl_list']),
	(CursorKind.DEFAULT_STMT,['stmt']),
	(CursorKind.DO_STMT,['stmt','expr']),
	(CursorKind.ENUM_CONSTANT_DECL,['expr_option']),
	(CursorKind.ENUM_DECL,['decl_list']),
	(CursorKind.FIELD_DECL,['var','type_list']), # [int hogehuga] in [struct{int hogehuga};]
	(CursorKind.FLOATING_LITERAL,['literal']),
	(CursorKind.FOR_STMT,['expr_option','expr_option','expr_option','stmt']),
	(CursorKind.FUNCTION_DECL,['type_list','stmt_option']), # TODO(satos) ayasii
	(CursorKind.GOTO_STMT,['label']),
	(CursorKind.IF_STMT,['expr','stmt','stmt_option']),
	(CursorKind.IMAGINARY_LITERAL,None), # TODO(satos) imaginary number. currently omit. 
	(CursorKind.INDIRECT_GOTO_STMT,None), # TODO(satos) like goto *hoge. Is this valid in standard C?
	(CursorKind.INIT_LIST_EXPR,['expr_list']), # TODO(satos) sometimes too long, restrict?
	(CursorKind.INTEGER_LITERAL,['literal']),
	(CursorKind.LABEL_REF,['label']),
	(CursorKind.LABEL_STMT,['label','stmt']),
	(CursorKind.MEMBER_REF,['var']), # x in {.x=3};
	(CursorKind.MEMBER_REF_EXPR,['expr']),
	(CursorKind.NULL_STMT,[]),
	(CursorKind.PACKED_ATTR,None), # __attribute__((packed))   skip. (or remove?)
	(CursorKind.PAREN_EXPR,['expr']),
	(CursorKind.PARM_DECL,['type_list']), # f(int,int)    argument type. 
	(CursorKind.PURE_ATTR,None), # __attribute__((__pure__))    skip. (or remove?)
	(CursorKind.RETURN_STMT,['expr_option']),
	(CursorKind.STRING_LITERAL,['literal']),
	(CursorKind.STRUCT_DECL,['decl_list']),
	(CursorKind.SWITCH_STMT,['expr','stmt']),
	(CursorKind.StmtExpr,None), # return ({int x = 3;4;});   GNU extension?, skip.
	(CursorKind.TRANSLATION_UNIT,['decl_list']),
	(CursorKind.TYPEDEF_DECL,['type_list']), #ayasii
	(CursorKind.TYPE_REF,['var']),
	(CursorKind.UNARY_OPERATOR,['unary_op','expr']), #what is difference with CursorKind.CXX_UNARY_EXPR
	(CursorKind.UNEXPOSED_ATTR,None), # __attribute__((__nothrow__))   skip. (or remove?)
	(CursorKind.UNEXPOSED_EXPR,['expr']), # Implicit cast (len==1) or initialize struct like .d=3 (len==2) or other insane expressions
	(CursorKind.UNION_DECL,['decl_list']),
	(CursorKind.VAR_DECL,['type_decl','expr_option']), # ayasii
	(CursorKind.VISIBILITY_ATTR,None), # __attribute__((visibility("hidden"))). skip.
	(CursorKind.WHILE_STMT,['expr','stmt']),
]

	
	
	
	
	




