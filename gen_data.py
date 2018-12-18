#! /usr/bin/env /home/satos/sotsuron/neural_decompiler/python_fix_seed.sh

from pycparser import c_parser, c_ast, parse_file
	
srcs = ['abspath', 'advice', 'alias', 'alloc', 'archive-tar', 'archive-zip', 'archive', 'argv-array', 'base85', 'bisect', 'blame', 'blob', 'branch', 'bulk-checkin', 'cache-tree', 'chdir-notify', 'check-racy', 'checkout', 'column', 'combine-diff', 'commit-graph', 'commit-reach', 'commit', 'common-main', 'connect', 'connected', 'convert', 'copy', 'credential-cache--daemon', 'credential-cache', 'credential-store', 'credential', 'csum-file', 'ctype', 'daemon', 'date', 'decorate', 'delta-islands', 'diff-delta', 'diff-lib', 'diff-no-index', 'diff', 'diffcore-break', 'diffcore-delta', 'diffcore-pickaxe', 'diffcore-rename', 'dir-iterator', 'editor', 'entry', 'environment', 'fast-import', 'fetch-negotiator', 'fetch-object', 'fetch-pack', 'fsck', 'fsmonitor', 'fuzz-pack-headers', 'fuzz-pack-idx', 'gpg-interface', 'graph', 'grep', 'hashmap', 'help', 'hex', 'http-backend', 'http-fetch', 'http-push', 'http-walker', 'http', 'ident', 'imap-send', 'interdiff', 'json-writer', 'kwset', 'levenshtein', 'line-log', 'line-range', 'linear-assignment', 'list-objects-filter-options', 'list-objects-filter', 'list-objects', 'll-merge', 'lockfile', 'ls-refs', 'mailinfo', 'match-trees', 'mem-pool', 'merge-blobs', 'merge-recursive', 'merge', 'mergesort', 'midx', 'name-hash', 'notes-cache', 'notes-merge', 'notes-utils', 'notes', 'object', 'oidmap', 'oidset', 'pack-bitmap-write', 'pack-bitmap', 'pack-check', 'pack-objects', 'pack-revindex', 'pack-write', 'packfile', 'parse-options-cb', 'parse-options', 'patch-delta', 'patch-ids', 'path', 'pathspec', 'pkt-line', 'preload-index', 'pretty', 'prio-queue', 'progress', 'prompt', 'protocol', 'quote', 'range-diff', 'reachable', 'read-cache', 'rebase-interactive', 'ref-filter', 'reflog-walk', 'refs', 'refspec', 'remote-curl', 'remote-testsvn', 'remote', 'replace-object', 'repository', 'rerere', 'resolve-undo', 'revision', 'send-pack', 'serve', 'server-info', 'setup', 'sh-i18n--envsubst', 'sha1-array', 'sha1-lookup', 'sha1-name', 'shallow', 'shell', 'sigchain', 'split-index', 'strbuf', 'streaming', 'string-list', 'sub-process', 'submodule-config', 'submodule', 'symlinks', 'tag', 'tempfile', 'thread-utils', 'tmp-objdir', 'trace', 'trailer', 'transport-helper', 'transport', 'tree-diff', 'tree-walk', 'tree', 'unix-socket', 'upload-pack', 'url', 'usage', 'utf8', 'varint', 'versioncmp', 'walker', 'wildmatch', 'worktree', 'wrapper', 'write-or-die', 'ws', 'xdiff-interface', 'zlib']

class ParseErr(Exception):
	def __init__(sl,s):
		pass

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

def parse_c(prsc):
	#print(s2tree(prsc))
	
	res = []
	tree = s2tree(prsc)
	#print(tree)
	def trav(d):
		nonlocal res
		ntp,ds = d
		nt,pos = ntp
		
		# CompoundStmt は多分飛ばさないといけない(if文のあととかがへんになる)
		if nt != 'CompoundStmt' and pos is not None:
			v = pos.split(':')
			a,b = int(v[1])-1,int(v[3])-1 #0-indexed !!
			#print(nt,pos,a,b)
			res.append((a,b))
		
		for x in ds:
			trav(x)
		
	trav(tree)
	
	"""
	lines = re.findall('<line:\d*:\d*, line:\d*:\d*>',prsc)
	#print(lines)
	#code.interact(local={'tree':tree})
	
	res = []
	for d in lines:
		v = d.split(':')
		#print(v)
		a,b = int(v[1])-1,int(v[3])-1 #0-indexed !!
		#print(a,b)
		res.append((a,b))
	"""
	
	return res
	
def get_line_from_list(asm,objd):
	asm = asm.split('\n')
	asm2 = filter(lambda x: x!='' and x[0]!='.' and x[:2] != '\t.',asm)
	asm2 = list(asm2)
	asm = filter(lambda x: x!='' and x[0]!='.' and (x[:2] != '\t.' or x[:5]=='\t.loc'),asm)
	asm = list(asm)
	
	objd = objd.split('\n')[5:]
	objd = filter(lambda x: x!='',objd)
	objd = list(objd)
	
	#print(len(asm2),len(objd))
	#print('\n'.join(asm2))
	#print('-' * 100)
	#print('\n'.join(objd))
	#assert len(objd) == len(asm2)
	if len(objd) != len(asm2):
		raise ParseErr("nanka hen")
	
	"""
	print(len(asm),len(objd))
	print('\n'.join(asm))
	print('-' * 100)
	print('\n'.join(objd))
	"""
	
	i = 0
	nl = ""
	res = []
	for s in asm:
		if s[:5]=='\t.loc':
			nl = s
		else:
			if s[-1]!=':':
				nld = nl.split(' ')[2]
				#print(nld,s,objd[i])
				#res.append((nld,objd[i]))
				
				#print(nld,' ',s)
				#print(nld,' ',objd[i].split('\t')[1])
				nbs = objd[i].split('\t')[1]
				nbs = filter(lambda x: x!='',nbs.split(' '))
				nbs = list(nbs)
				#print(nld,nbs)
				res.append((int(nld)-1,nbs)) #1-indexed!!
			i += 1
	return res


data = []
vocab_src = []

import collections 

#srcs = ['cat']
for fn in srcs:
	fn = './build/' + fn
	with open(fn + '.s') as fp:
		asm = fp.read()
	with open(fn + '.objd') as fp:
		objd = fp.read()
	try:
		ds = get_line_from_list(asm,objd)
	except ParseErr as s:
		#print(s)
		continue
	#print(ds)
	with open(fn + '.parsed') as fp:
		parsed = fp.read()
	#print(fn)
	lines = parse_c(parsed)
	#print(ds)
	#print(lines)
	#continue
	
	with open(fn + '.tokenized.c') as fp:
		csrc = fp.read()
		csrc = csrc.split('\n')

	with open(fn + '.tokenized') as fp:
		csrc_with_type = list(map(lambda x: x.split(' ')[0],fp.read().split('\n')))
	
	#print(fn)
	for a,b in lines:
		nas = []
		for i,d in ds:
			if a <= i and i < b:
				nas += d
				#nas += 
		if len(nas)==0:
			continue
		
		
		def i2srcdata(i):
			if 'string_literal' == csrc_with_type[i]:
				return '__STRING__'
			return csrc[i]
		
		ncs = [i2srcdata(i) for i in range(a,b+1)]
		#ncs = csrc[a:b+1]

		vocab_src += ncs
		
		data.append((nas,ncs))


vocab_src = sorted(map(lambda xy: (xy[1],xy[0]),collections.Counter(vocab_src).items()))[::-1][:252]
lvoc = len(vocab_src)
vocab_src = list(map(lambda xy: xy[1],vocab_src))
#print(vocab_src)
#exit()
data = list(map(lambda xy: (list(map(lambda t: int(t,16),xy[0])),list(map(lambda t: vocab_src.index(t) if t in vocab_src else lvoc,xy[1]))),data))
vocab_src.append('__SOME__')

#data = list(map(lambda xy: (list(map(lambda t: int(t,16),xy[0])),xy[1]),data))

print('data = ' + str(data))
print('src_bocab = ' + str(vocab_src))  





