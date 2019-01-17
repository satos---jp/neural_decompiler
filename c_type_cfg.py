
"""
type_exps = {<class 'pycparser.c_ast.Typename'>: {('name', 'quals', 'type')}, <class 'pycparser.c_ast.FuncDecl'>: {('args', 'type')}, <class 'pycparser.c_ast.ParamList'>: {('params',)}, <class 'pycparser.c_ast.TypeDecl'>: {('declname', 'quals', 'type')}, <class 'pycparser.c_ast.Struct'>: {('name', 'decls')}, <class 'pycparser.c_ast.IdentifierType'>: {('names',)}, <class 'pycparser.c_ast.PtrDecl'>: {('quals', 'type')}, <class 'pycparser.c_ast.Union'>: {('name', 'decls')}, <class 'pycparser.c_ast.Enum'>: {('name', 'values')}, <class 'pycparser.c_ast.ArrayDecl'>: {('type', 'dim', 'dim_quals')}, <class 'pycparser.c_ast.Constant'>: {('type', 'value')}, <class 'pycparser.c_ast.ID'>: {('name',)}, <class 'pycparser.c_ast.BinaryOp'>: {('op', 'left', 'right')}, <class 'pycparser.c_ast.TernaryOp'>: {('cond', 'iftrue', 'iffalse')}, <class 'pycparser.c_ast.StructRef'>: {('name', 'type', 'field')}, <class 'pycparser.c_ast.UnaryOp'>: {('op', 'expr')}}
"""

"""
type_exps = {
	pycparser.c_ast.Typename: ('name', 'quals', 'type'),
	pycparser.c_ast.FuncDecl: ('args', 'type'), 
	pycparser.c_ast.ParamList: {('params',)}, <class 'pycparser.c_ast.TypeDecl'>: {('declname', 'quals', 'type')}, <class 'pycparser.c_ast.Struct'>: {('name', 'decls')}, <class 'pycparser.c_ast.IdentifierType'>: {('names',)}, <class 'pycparser.c_ast.PtrDecl'>: {('quals', 'type')}, <class 'pycparser.c_ast.Union'>: {('name', 'decls')}, <class 'pycparser.c_ast.Enum'>: {('name', 'values')}, <class 'pycparser.c_ast.ArrayDecl'>: {('type', 'dim', 'dim_quals')}, <class 'pycparser.c_ast.Constant'>: {('type', 'value')}, <class 'pycparser.c_ast.ID'>: {('name',)}, <class 'pycparser.c_ast.BinaryOp'>: {('op', 'left', 'right')}, <class 'pycparser.c_ast.TernaryOp'>: {('cond', 'iftrue', 'iffalse')}, <class 'pycparser.c_ast.StructRef'>: {('name', 'type', 'field')}, <class 'pycparser.c_ast.UnaryOp'>: {('op', 'expr')}}
"""




"""
{<class 'pycparser.c_ast.IdentifierType'>: {(((<class 'list'>, <class 'str'>), 'names'),)}, <class 'pycparser.c_ast.TypeDecl'>: {((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.Enum'>, 'type')), ((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.Struct'>, 'type')), ((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.Union'>, 'type')), ((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.Enum'>, 'type')), ((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.IdentifierType'>, 'type')), ((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.IdentifierType'>, 'type')), ((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.Union'>, 'type')), ((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.Struct'>, 'type'))}, <class 'pycparser.c_ast.Typename'>: {((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), ((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type')), ((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), ((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), ((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type')), ((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), ((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), ((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type'))}, <class 'pycparser.c_ast.Struct'>: {((<class 'str'>, 'name'), (<class 'NoneType'>, 'decls'))}, <class 'pycparser.c_ast.PtrDecl'>: {(((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), ((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), (((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), ((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), (((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type')), (((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), ((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), ((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type'))}, <class 'pycparser.c_ast.ParamList'>: {(((<class 'list'>, <class 'pycparser.c_ast.Typename'>), 'params'),)}, <class 'pycparser.c_ast.FuncDecl'>: {((<class 'pycparser.c_ast.ParamList'>, 'args'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), ((<class 'NoneType'>, 'args'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), ((<class 'pycparser.c_ast.ParamList'>, 'args'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), ((<class 'NoneType'>, 'args'), (<class 'pycparser.c_ast.PtrDecl'>, 'type'))}, <class 'pycparser.c_ast.Union'>: {((<class 'str'>, 'name'), (<class 'NoneType'>, 'decls'))}, <class 'pycparser.c_ast.Constant'>: {((<class 'str'>, 'type'), (<class 'str'>, 'value'))}, <class 'pycparser.c_ast.ArrayDecl'>: {((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.StructRef'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'NoneType'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'pycparser.c_ast.Constant'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.Constant'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'pycparser.c_ast.Constant'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.TernaryOp'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'pycparser.c_ast.StructRef'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.BinaryOp'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'pycparser.c_ast.ID'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'pycparser.c_ast.BinaryOp'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.ID'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'NoneType'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'pycparser.c_ast.ID'>, 'dim'), (<class 'list'>, 'dim_quals')), ((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'NoneType'>, 'dim'), (<class 'list'>, 'dim_quals'))}, <class 'pycparser.c_ast.Enum'>: {((<class 'str'>, 'name'), (<class 'NoneType'>, 'values'))}, <class 'pycparser.c_ast.ID'>: {((<class 'str'>, 'name'),)}, <class 'pycparser.c_ast.BinaryOp'>: {((<class 'str'>, 'op'), (<class 'pycparser.c_ast.UnaryOp'>, 'left'), (<class 'pycparser.c_ast.Constant'>, 'right')), ((<class 'str'>, 'op'), (<class 'pycparser.c_ast.Constant'>, 'left'), (<class 'pycparser.c_ast.ID'>, 'right')), ((<class 'str'>, 'op'), (<class 'pycparser.c_ast.BinaryOp'>, 'left'), (<class 'pycparser.c_ast.Constant'>, 'right')), ((<class 'str'>, 'op'), (<class 'pycparser.c_ast.Constant'>, 'left'), (<class 'pycparser.c_ast.UnaryOp'>, 'right')), ((<class 'str'>, 'op'), (<class 'pycparser.c_ast.ID'>, 'left'), (<class 'pycparser.c_ast.ID'>, 'right')), ((<class 'str'>, 'op'), (<class 'pycparser.c_ast.ID'>, 'left'), (<class 'pycparser.c_ast.Constant'>, 'right'))}, <class 'pycparser.c_ast.StructRef'>: {((<class 'pycparser.c_ast.ID'>, 'name'), (<class 'str'>, 'type'), (<class 'pycparser.c_ast.ID'>, 'field'))}, <class 'pycparser.c_ast.EllipsisParam'>: {()}, <class 'pycparser.c_ast.TernaryOp'>: {((<class 'pycparser.c_ast.BinaryOp'>, 'cond'), (<class 'pycparser.c_ast.ID'>, 'iftrue'), (<class 'pycparser.c_ast.ID'>, 'iffalse'))}, <class 'pycparser.c_ast.UnaryOp'>: {((<class 'str'>, 'op'), (<class 'pycparser.c_ast.ID'>, 'expr'))}}
"""


"""
type_expr = {
	pycparser.c_ast.Typename: [
		((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), 
		((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type')), 
		((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), 
		((<class 'NoneType'>, 'name'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), 
		((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type')), 
		((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), 
		((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), 
		((<class 'NoneType'>, 'name'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')),
	],

	pycparser.c_ast.ArrayDecl: [
		((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.StructRef'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'NoneType'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'pycparser.c_ast.Constant'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.Constant'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'pycparser.c_ast.Constant'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.TernaryOp'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'pycparser.c_ast.StructRef'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.BinaryOp'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'pycparser.c_ast.ID'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'pycparser.c_ast.BinaryOp'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'pycparser.c_ast.ID'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.ArrayDecl'>, 'type'), (<class 'NoneType'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.PtrDecl'>, 'type'), (<class 'pycparser.c_ast.ID'>, 'dim'), (<class 'list'>, 'dim_quals')), 
		((<class 'pycparser.c_ast.TypeDecl'>, 'type'), (<class 'NoneType'>, 'dim'), (<class 'list'>, 'dim_quals'))
	],
	
	pycparser.c_ast.TypeDecl: [
		((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.Enum'>, 'type')), 
		((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.Struct'>, 'type')), 
		((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.Union'>, 'type')), 
		((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.Enum'>, 'type')), 
		((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.IdentifierType'>, 'type')), 
		((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.IdentifierType'>, 'type')), 
		((<class 'NoneType'>, 'declname'), (<class 'list'>, 'quals'), (<class 'pycparser.c_ast.Union'>, 'type')), 
		((<class 'NoneType'>, 'declname'), ((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.Struct'>, 'type'))
	],
	
	pycparser.c_ast.PtrDecl: [
		(((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), 
		((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), 
		(((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), 
		((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.ArrayDecl'>, 'type')), 
		(((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type')), 
		(((<class 'list'>, <class 'str'>), 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), 
		((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), 
		((<class 'list'>, 'quals'), (<class 'pycparser.c_ast.FuncDecl'>, 'type'))
	],
	
	pycparser.c_ast.FuncDecl: [
		((<class 'pycparser.c_ast.ParamList'>, 'args'), (<class 'pycparser.c_ast.PtrDecl'>, 'type')), 
		((<class 'NoneType'>, 'args'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), 
		((<class 'pycparser.c_ast.ParamList'>, 'args'), (<class 'pycparser.c_ast.TypeDecl'>, 'type')), 
		((<class 'NoneType'>, 'args'), (<class 'pycparser.c_ast.PtrDecl'>, 'type'))
	], 
	
	pycparser.c_ast.ParamList: [
		(((<class 'list'>, <class 'pycparser.c_ast.Typename'>), 'params'),)
	],
}
"""

import pycparser

type_root = pycparser.c_ast.Typename

nont_reduce = {
	'type': [
		pycparser.c_ast.ArrayDecl,
		pycparser.c_ast.FuncDecl,
		pycparser.c_ast.TypeDecl,
		pycparser.c_ast.PtrDecl,
	],
	'paramlist': [
		pycparser.c_ast.ParamList
	],
	'typename': [
		pycparser.c_ast.Typename
	],
	'typedecltype': [
		pycparser.c_ast.Struct,
		pycparser.c_ast.Union,
		pycparser.c_ast.IdentifierType,
	],
}

cfg = {
	pycparser.c_ast.Typename: [(('string','list'),'quals'),('type','type')],
	
	pycparser.c_ast.ArrayDecl: [(('expr','option'),'dim'),('type','type')],
	pycparser.c_ast.TypeDecl: [(('string','list'),'quals'),('typedecltype','type')],
	pycparser.c_ast.PtrDecl: [(('string','list'),'quals'),('type','type')],
	pycparser.c_ast.FuncDecl: [(('paramlist','option'),'args'),('type','type')],
	pycparser.c_ast.ParamList: [(('typename','list'),'params')],
	
	pycparser.c_ast.Struct: [('string','type_name')],
	pycparser.c_ast.Union: [('string','type_name')],
	pycparser.c_ast.IdentifierType: [(('string','list'),'names')],
}



