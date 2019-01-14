class HelpAST:
	def __init__(sl,tag,data=None):
		sl.tag = tag
		sl.data = data
		sl.cs = []

class ParseErr(Exception):
	def __init__(sl,s):
		pass

def s2tree(s):
	s = s.split('\n')
	ls = len(s)
	i = 0
	def parse(d):
		nonlocal i,s,ls
		ds = s[i][d:].split(' ')
		tag = ds[0]
		data = ds[1:]
		if len(data)>1 and data[1][0]=='<' and data[1][-1]!='>':
			data = [data[0],data[1]+data[2]] + data[3:]
		if len(data)>2 and data[2][0]=='<' and data[2][-1]!='>':
			data = data[:2] + [data[2]+data[3]] + data[4:]
		i += 1
		if tag in ['<<<NULL>>>']:
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
				print(s[i][d:])
				raise ParseErr('miss at %d:%d' % (i,d))
		
		
		node.cs = cs
		if tag in ['array']:
			return None
		else:
			return node
	
	return parse(0)

import os
def fn_to_clang_ast(fn):
	os.system('clang -Xclang -ast-dump -fsyntax-only %s > %s.astdump 2>/dev/null' % (fn,fn))
	with open(fn + '.astdump') as fp:
		parsed = fp.read()
	return s2tree(parsed)









