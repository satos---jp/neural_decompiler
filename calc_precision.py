import edit_distance
import pickle

from nltk.translate import bleu_score

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

	ds = [(a,normalize_sentense(b),normalize_sentense(c)) for (a,b),c in ds]

	edit_ratio = 0
	bleu_ratio = 0
	for a,b,c in ds:	
		assert type(b) is list
		assert type(c) is list
		#print(b)
		#print(c)
		edit_ratio += 1.0 - edit_distance.SequenceMatcher(b,c).ratio()
		bleu_ratio += bleu_score.sentence_bleu([b],c,smoothing_function=bleu_score.SmoothingFunction().method1)
	edit_ratio /= len(ds)
	bleu_ratio /= len(ds)
	return (edit_ratio,bleu_ratio)
	

def calc_ast(fn):
	with open(fn,'rb') as fp:
		ds = pickle.load(fp)
	#print(len(ds))
	ds = ds[:1000]
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
	
	ds = [(a,tree2normalizedsentense(b),tree2normalizedsentense(c)) for (a,b),c in ds]
	ds = list(filter(lambda abc: (abc[1],abc[2]) != ([],[]),ds))
	edit_ratio = 0
	bleu_ratio = 0
	for a,b,c in ds:	
		assert type(b) is list
		assert type(c) is list
		#print(' '.join(b))
		#print(' '.join(c))
		#print('-' * 100)
		edit_ratio += 1.0 - edit_distance.SequenceMatcher(b,c).ratio()
		bleu_ratio += bleu_score.sentence_bleu([b],c,smoothing_function=bleu_score.SmoothingFunction().method1)
	edit_ratio /= len(ds)
	bleu_ratio /= len(ds)
	return (edit_ratio,bleu_ratio)
	
#print(calc_ast('o4.pickle'))
#exit()
#print(calc_ast('translate_data/seq2tree_flatten/iter_1800_transdata.pickle'))
#exit()

"""
print('[')
for i in range(65):
	fn = 'translate_data/seq2seq/iter_%d_transdata.pickle' % (i*100)
	r = calc_csrc(fn)
	print(r,',')
print(']')


print('[')
for i in range(21):
	fn = 'translate_data/seq2seq_withatt/iter_%d_transdata.pickle' % (i*100)
	r = calc_csrc(fn)
	print(r,',')
print(']')
"""

print('[')
for i in range(19):
	fn = 'translate_data/seq2tree_flatten_verbose_choice/iter_%d_transdata.pickle' % (i*100)
	r = calc_ast(fn)
	print(r,',')
print(']')
"""
print('[')
for i in range(21):
	fn = 'translate_data/seq2tree/iter_%d_transdata.pickle' % (i*100)
	r = calc_ast(fn)
	print(r,',')
print(']')
"""
