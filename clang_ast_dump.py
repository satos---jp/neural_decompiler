class HelpAST:
	def __init__(sl,tag,data=None):
		sl.tag = tag
		sl.data = data
		sl.cs = []

def s2tree(s):
	s = s.split('\n')
	ls = len(s)
	i = 0
	def parse(d):
		nonlocal i,s,ls
		ds = s[i][d:].split(' ')
		tag = ds[0]
		data = ds[1:]
		i += 1
		if tag == '<<<NULL>>>':
			return None
		node = HelpAST(tag,data)
		if i >= ls or len(s[i])<=d or (s[i][d]!='|' and s[i][d]!='`'):
			return node
		
		cs = []
		while True:
			if s[i][d:d+2]=='|-':
				cs.append(parse(d+2))
			elif s[i][d:d+2]=='`-':
				cs.append(parse(d+2))
				break
			else:
				#return res
				#print(s[i][d:])
				raise ParseErr('miss at %d:%d' % (i,d))
		
		
		node.cs = cs
		return node
	
	return parse(0)

import os
def fn_to_clang_ast(fn):
	os.system('clang -Xclang -ast-dump -fsyntax-only %s > %s.astdump 2>/dev/null' % (fn,fn))
	with open(fn + '.astdump') as fp:
		parsed = fp.read()
	return s2tree(parsed)









