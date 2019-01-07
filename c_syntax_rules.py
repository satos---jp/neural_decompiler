#! /usr/bin/env /home/satos/sotsuron/neural_decompiler/python_fix_seed.sh
	
class ParseErr(Exception):
	def __init__(sl,s):
		pass

# syntax is part of
# http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1548.pdf 
# annex A (page 454)
# A.2 about Phrase structure grammer (page 461~) 
# Skip: 
# generic-selection
"""
toplevel = decl*

compound-statement

"""







def s2tree(s):
	s = s.split('\n')
	ls = len(s)
	i = 0
	def parse(d):
		nonlocal i,s,ls
		node = s[i][d:]
		nt = node.split(' ')[0]
		#print(node)
		#try:
		ps = re.findall('<line:\d*:\d, line:\d*:\d*>',node)
		if len(ps)>0:
			pos = ps[0]
		else:
			pos = None
		
		node = (nt,pos)
		
		i += 1
		#print(i,d)
		#print(s[i])
		if i >= ls or len(s[i])<=d or (s[i][d]!='|' and s[i][d]!='`'):
			#print(node)
			return (node,[])
		
		res = []
		while True:
			if s[i][d:d+2]=='|-':
				res.append(parse(d+2))
			elif s[i][d:d+2]=='`-':
				res.append(parse(d+2))
				break
			else:
				#return res
				#print(s[i][d:])
				raise ParseErr('miss at %d:%d' % (i,d))
		return (node,res)
	
	return parse(0)

import code
import re


"""
rtc = {
	'expr':[
		'ArraySubscriptExpr','BinaryOperator',
		'CStyleCastExpr','ImplicitCastExpr',
		'IntegerLiteral','ParenExpr','CallExpr',
		'FloatingLiteral','UnaryOperator',
		
	],
}


cfg = {
	'BinaryOperator':('expr','expr'),
	'ArraySubscriptExpr':('expr','expr'),
	'BreakStmt':(),
	'CStyleCastExpr':('expr'),
	'CallExpr':('expr_list'),
	'CaseStmt':('expr',None,'stmt'),
	'CharacterLiteral':(),
	'CompoundStmt':('expr_stmt_list'),
	'ConditionalOperator':('expr','expr','expr'),
	'ConstAttr':(),
}
"""


# とりあえず、 expr か stmt の2択に落とす。


rules = {}

def parse_c(prsc):
	#print(s2tree(prsc))
	
	res = []
	tree = s2tree(prsc)
	#print(tree)
	def trav(v):
		global rules
		nonlocal res
		ntp,ds = v
		nt,pos = ntp
		
		ntrans = tuple(map(lambda x: x[0][0],ds))
		if nt in rules.keys():
			rules[nt] |= set([ntrans])
		else:
			rules[nt] = set([ntrans])
		
		# CompoundStmt は多分飛ばさないといけない(if文のあととかがへんになる)
		if nt != 'CompoundStmt' and pos is not None:
			x = pos.split(':')
			a,b = int(x[1])-1,int(x[3])-1 #0-indexed !!
			#print(nt,pos,a,b)
			res.append((a,b))
		
		for x in ds:
			trav(x)
		
	trav(tree)
	
	return res


class InvalidToken(Exception):
	pass


import collections 
import os
import sys
import random

srcs = list(map(lambda x: x[:-5],filter(lambda x: x[-5:]=='.objd',os.listdir('./build'))))
#srcs = ['cat']
print(len(srcs))


srcs = srcs
for fn in srcs[:10]:
	
	fn = './build/' + fn
	print(fn)
	with open(fn + '.parsed') as fp:
		parsed = fp.read()
	#print(fn)
	parse_c(parsed)
	#print(ds)
	#print(lines)
	#continue


for k,v in rules.items():
	lenvs = set(map(lambda x: len(x),v))
	#print(k,lenvs)
	if len(lenvs) == 1:
		pass
	elif len(lenvs) == 1:
		#print(k,lenvs)
		pass
	else:
		#print(k,lenvs)
		pass
	#print(k,v)










