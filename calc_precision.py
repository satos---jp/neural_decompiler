import edit_distance
import pickle

import csrc2ast
	
import sys
sys.setrecursionlimit(10000)	

with open('dataset_asm_csrc/vocab_csrc.pickle','rb') as fp:
	c_vocab = pickle.load(fp)


def calc_csrc(fn):
	with open(fn,'rb') as fp:
		ds = pickle.load(fp)
	def normalize_sentense(s):
		s = list(map(lambda i: c_vocab[i],s))
		ts = []
		names = {}
		for d in s:
			if d[:5]=='__ID_':
				if not d in names.keys():
					names[d] = '__NORMALIZE_TMP_ID_%d__' % len(names.keys())
				d = names[d]
			ts.append(d)
		return ts
	
	ratio = 0
	for (a,b),c in ds:	
		b = normalize_sentense(b)
		c = normalize_sentense(c)
		assert type(b) is list
		assert type(c) is list
		print(b)
		print(c)
		ratio += 1.0 - edit_distance.SequenceMatcher(b,c).ratio()
	ratio /= len(ds)
	return ratio
	

def calc_ast(fn):
	with open(fn,'rb') as fp:
		ds = pickle.load(fp)
	def tree2normalizedsentense(tree,normalize=True):
		s = csrc2ast.ast2token_seq(tree)
		if not normalize:
			return s
		ts = []
		names = []
		for sd in s:
			if sd[:8]=='__ALPHA_':
				if not sd in names:
					names.append(sd)
				sd = 'VAR_%d' % names.index(sd)
			ts.append(sd)
		return ts
	
	ratio = 0
	for (a,b),c in ds:	
		b = tree2normalizedsentense(b)
		c = tree2normalizedsentense(c)
		assert type(b) is list
		assert type(c) is list
		print(' '.join(b))
		print(' '.join(c))
		print('-' * 100)
		ratio += 1.0 - edit_distance.SequenceMatcher(b,c).ratio()
	ratio /= len(ds)
	return ratio
	
#print(calc_csrc('o.pickle'))
print(calc_ast('translate_data/seq2tree_flatten/iter_1800_transdata.pickle'))
exit()

for i in range(19):
	fn = 'translate_data/seq2tree_flatten/iter_%d_transdata.pickle' % (i*100)
	r = calc(fn)
	print(fn,r)
