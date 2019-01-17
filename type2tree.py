from pycparser import c_parser,c_ast,plyparser

parser = c_parser.CParser()

import code


kvs = {}
def ty2tree(ty):
	try:
		ki = parser.parse('int _ = (' + ty + ')0;', filename='<stdin>')
	except plyparser.ParseError:
		return
	ki = ki.ext[0].init.to_type
	#print(type(ki))
	#return
	#code.interact(local={'ki':ki})
	
	def f(t):
		if type(t) is c_ast.TypeDecl:
			print(t)
			code.interact(local={'t':t})
		attrs = list(filter(lambda x: x not in ['coord', '__weakref__'], list(t.__slots__)))
		#print(type(t),attrs)
		for d in attrs:
			nv = t.__getattribute__(d)
			if type(nv).__module__ == 'pycparser.c_ast':
				f(nv)
			elif type(nv) is list:
				for knv in nv:
					if type(knv).__module__ == 'pycparser.c_ast':
						f(knv)
		
		
		def g(v):
			rt = type(v)
			if type(v) is list and len(v)>0:
				rt = (rt,type(v[0]))
			return rt
		
		tt = type(t)
		attrs_types = list(map(lambda x: (g(t.__getattribute__(x)),x),attrs))
		if tt in kvs.keys():
			kvs[tt] |= set([tuple(attrs_types)])
		else:
			kvs[tt] = set([tuple(attrs_types)])

	f(ki)
#def tree2tys(ty):
#	


if __name__=='__main__':
	import pickle
	with open('node_expand_data.pickle','rb') as fp:
		d = pickle.load(fp)
	for t in list(d['type']):
		ty2tree(t)
	print(kvs)
