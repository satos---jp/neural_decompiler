#import edit_distance
import pickle

from nltk.translate import bleu_score

import c_cfg
import csrc2ast
import code
import sys
sys.setrecursionlimit(10000)	

with open('dataset_asm_csrc/vocab_csrc.pickle','rb') as fp:
	c_vocab = pickle.load(fp)

# verified by http://judge.u-aizu.ac.jp/onlinejudge/review.jsp?rid=3362979#1
def my_edit_dist(s,t):
	#print('call',s,t)
	mem = {}
	def dp(i,j):
		nonlocal mem,s,t
		if (i,j) in mem.keys():
			return mem[(i,j)]
		if i<0:
			res = j+1
		elif j<0:
			res = i+1
		else:
			res = min(dp(i-1,j)+1,dp(i,j-1)+1,dp(i-1,j-1) + (0 if s[i]==t[j] else 1))
		mem[(i,j)] = res
		#print(i,j,s[:i+1],t[:j+1],res)
		return res
	 
	return dp(len(s)-1,len(t)-1)


def calc_edit_ratio(s,t):
	#edit_distance.SequenceMatcher(s,t).ratio()
	return my_edit_dist(s,t)/max(len(s),len(t))

def calc_csrc(fn):
	#print(fn)
	with open(fn,'rb') as fp:
		ds = pickle.load(fp)
	ds = ds[:100]
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
		b = list(map(lambda x: x.replace('__NORMALIZE_TMP_ID_','V'),b))
		b = list(map(lambda x: x.replace('__',''),b))
		c = list(map(lambda x: x.replace('__NORMALIZE_TMP_ID_','V'),c))
		c = list(map(lambda x: x.replace('__',''),c))
		#print(' '.join(b))
		#print(' '.join(c))
		#print('-' * 100)
		edit_ratio += calc_edit_ratio(b,c)
		bleu_ratio += bleu_score.sentence_bleu([b],c,smoothing_function=bleu_score.SmoothingFunction().method1)
	edit_ratio /= len(ds)
	bleu_ratio /= len(ds)
	return (edit_ratio,bleu_ratio)
	

def calc_ast(fn):
	#print(fn)
	with open(fn,'rb') as fp:
		ds = pickle.load(fp)
	#print(len(ds))
	ds = ds[:100]
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
				sd = 'V%d' % names.index(sd)
			ts.append(sd)
		return ts
	#code.interact(local={'ds':ds})
	#c_cfg.data_miyasui(ds[0][1][2][0])
	#exit()
	#print(ds[0][-1][-1])
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
		edit_ratio += calc_edit_ratio(b,c)
		bleu_ratio += bleu_score.sentence_bleu([b],c,smoothing_function=bleu_score.SmoothingFunction().method1)
	edit_ratio /= len(ds)
	bleu_ratio /= len(ds)
	return (edit_ratio,bleu_ratio)
	
#print(calc_ast('o4.pickle'))
#exit()
#print(calc_ast('translate_data/seq2tree_flatten/iter_1800_transdata.pickle'))
#exit()


#calc_csrc('sample_seq2seq.pickle')
#calc_csrc('sample_seq2seq_withatt.pickle')
#calc_ast('sample_seq2tree_flatten.pickle')


print('seq2seq = [')
for i in range(65):
	fn = 'translate_data/seq2seq/iter_%d_transdata.pickle' % (i*100)
	r = calc_csrc(fn)
	print(r,',')
print(']')

print('seq2seq_att = [')
for i in range(21):
	fn = 'translate_data/seq2seq_withatt/iter_%d_transdata.pickle' % (i*100)
	r = calc_csrc(fn)
	print(r,',')
print(']')

print('seq2tree_flatten = [')
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
