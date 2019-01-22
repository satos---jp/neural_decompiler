#! /usr/bin/python3
#coding: utf-8

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers

from model_rnn import Seq2seq
from model_rnn_with_att import Seq2seq_with_Att
from model_seq2tree import Seq2Tree
from model_seq2tree_flatten import Seq2Tree_Flatten


def convert(batch, device):
	def to_device_batch(batch):
		return batch

	return {'xs': to_device_batch([x for x,_ in batch]),
			'ys': to_device_batch([y for _,y in batch])}

import datetime
def getstamp():
	return datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

import code
import edit_distance
import random
import pickle
import csrc2ast
import c_cfg



def init_seq2seq():
	n_layer = 4
	n_unit = 64
	
	n_maxlen = 128
	
	with open('data_with_gcc_binary.pickle','rb') as fp:
			(data,c_vocab) = pickle.load(fp)
			asm_vocab = [i for i in range(256)]
	
	#with open('data_with_asmstring.pickle','rb') as fp:
	#		(data,c_vocab,asm_vocab) = pickle.load(fp)
	
	#data.data = list(filter(lambda xy: len(xy[0])<20 and len(xy[1])<10,data.data))
	data = list(filter(lambda xy: len(xy[1])<n_maxlen,data))
	data = data[:1000000]
	print(len(data))
	
	lds = len(data)
	train_data = data[:int(lds*0.8)]
	test_data = data[int(lds*0.8):]
	
	#source_ids = load_vocabulary(args.SOURCE_VOCAB)
	#target_ids = load_vocabulary(args.TARGET_VOCAB)
	maxlen_asm = max(map(lambda xy: len(xy[0]),data)) + 1
	maxlen_c   = max(map(lambda xy: len(xy[1]),data)) + 1
	print(maxlen_asm,maxlen_c)
	
	v_eos_src = len(asm_vocab)
	asm_vocab += ['__EOS__']
	src_vocab_len = len(asm_vocab)
	
	v_eos_dst = len(c_vocab)
	c_vocab += ['__EOS__']
	dst_vocab_len = len(c_vocab)
 
	model = Seq2seq(n_layer, src_vocab_len, dst_vocab_len , n_unit, v_eos_src, v_eos_dst, n_maxlen)
	#serializers.load_npz('models/iter_5600_editdist_0.866712_time_2019_01_15_04_50_45.npz',model)
	#serializers.load_npz('save_models/models_prestudy_edit_dist_0.65/iter_47600__edit_dist_0.691504__time_2018_12_28_01_36_30.npz',model)
	
	
	#ddata  = list(map(lambda xy: np.array(xy[0] + (maxlen_asm - len(xy[0])) * [v_eos],np.int32),data.data))
	#dlabel = list(map(lambda xy: np.array(xy[1] + (maxlen_src - len(xy[1])) * [v_eos],np.int32),data.data))
	ddata  = list(map(lambda xy: np.array(xy[0] + [v_eos_src],np.int32),train_data))
	dlabel = list(map(lambda xy: np.array(xy[1] + [v_eos_dst],np.int32),train_data))
	#dataset = chainer.datasets.tuple_dataset.TupleDataset(ddata,dlabel)
	dataset = list(zip(ddata,dlabel))
	#code.interact(local={'data': dataset})


	
	def normalize_sentense(s):
		ts = []
		names = {}
		for d in s:
			if d[:5]=='__ID_':
				if not sd in names.keys():
					names[d] = '__NORMALIZE_TMP_ID_%d__' % len(names.keys())
				d = names[d]
			ts.append(d)
		return ts
	
	logt = 0
	
	def calcurate_edit_distance():
		ratio = 0.0
		ds = random.sample(test_data, 100)
		for fr,to in ds:
			result = model.translate([model.xp.array(fr)])[0]
			result = list(result)
			result = normalize_sentense(result)
			to = normalize_sentense(to)
			ratio += 1.0 - edit_distance.SequenceMatcher(to,result).ratio()
		#print(ratio)
		ratio /= len(ds)
		return ratio
	
	@chainer.training.make_extension()
	def translate(trainer):
		eds = calcurate_edit_distance()
		print('edit distance:',eds)
		nonlocal logt
		logt += train_iter_step
		with open('log/iter_%s_editdist_%f_time_%s.txt' % (logt,eds,getstamp()),'w') as fp:
			res = ""
			for _ in range(10):
				source, target = test_data[np.random.choice(len(test_data))]
				result = model.translate([model.xp.array(source)])[0]
				
				#target = normalize_sentense(target)
				#result = normalize_sentense(result)
				
				#source_sentence = ' '.join([asm_vocab[x] for x in source])
				source_sentence = ' '.join([str(x) for x in source])
				target_sentence = ' '.join(csrc2ast.ast2token_seq(target))
				result_sentence = ' '.join(csrc2ast.ast2token_seq(result))
				res += ('# source : ' + source_sentence) + "\n"
				res += ('# expect : ' + target_sentence) + "\n"
				#print('# result_array : ' + ' '.join(["%d" % y for y in result]))
				res += ('# result : ' + result_sentence) + "\n"
			fp.write(res)
		
		serializers.save_npz('models/iter_%s_editdist_%f_time_%s.npz' % (logt,eds,getstamp()),model)
		
		#import code
		#code.interact(local={'d':model.embed_x})
		#print('embed data : ', model.embed_x.W)

	return (model,dataset,translate)





class Asm_ast_dataset(chainer.dataset.DatasetMixin):
	def __init__(self):
		super(Asm_ast_dataset,self).__init__()
		
		self.len = 1000190
		#self.len = 3200
		self.cache = []
		self.data_idx = []
		
		self.get_example = self.data_range(0,100)
		self.get_test = self.data_range(100,120)
		
	def __len__(self):
		return self.len
	
	def data_range(self,a,b):
		def gex(_):
			if self.cache == []:
				if self.data_idx == []:
					self.data_idx =list(range(a,b))
					random.shuffle(self.data_idx)
			
				fn = self.data_idx.pop()
				with open('dataset_asm_ast/data_sampled_%d.pickle' % fn,'rb') as fp:
					self.cache = pickle.load(fp)
					#self.cache = list(map(lambda xy: (np.array(xy[0],np.int32),xy[2]),self.cache))
					random.shuffle(self.cache)
					
			res = self.cache.pop()
			p,size,q = res
			#print(size)
			return (np.array(p,np.int32),q)
			#return (np.array([0],np.int32),(c_cfg.rootidx,0,[(0,0,[])]))
		
		return gex

def load_dataset():
	
	with open('dataset_asm_ast/c_trans_array.pickle','rb') as fp:
		c_syntax_arr = pickle.load(fp)
	with open('dataset_asm_ast/vocab_asm.pickle','rb') as fp:
		asm_vocab = pickle.load(fp)
	"""
	data = []
	for i in range(13):
		with open('dataset_asm_ast/data_%d.pickle' % i,'rb') as fp:
			data += pickle.load(fp)
	"""
	#with open('sampled_asm_ast_dataset.pickle','rb') as fp:
	#	res =  pickle.load(fp)
	
	res = (Asm_ast_dataset(),asm_vocab,c_syntax_arr)
	return res

def csize(d):
	res = 1
	_,_,cs = d
	for c in cs:
		res += csize(c)
	return res

import my_c_ast
train_iter_step = None
def init_seq2tree():
	global train_iter_step
	n_layer = 4
	n_unit = 128
	
	(data,asm_vocab,c_syntax_arr) = load_dataset()
	print('load dataset')
	"""
	for _,s,d in data:
		if s>=250:
			nast = my_c_ast.load_astdata(d)
			nast.before_show()
			print(nast.show())
			print('size:',s,'csize:',csize(d))
			exit()
	exit()
	"""
	#print(data[0])
	#data.data = list(filter(lambda xy: len(xy[0])<20 and len(xy[1])<10,data.data))
	#data = list(filter(lambda xy: len(xy[1])<n_maxsize,data))
	#data = data[:1000000]
	print(len(data))
	
	#lds = len(data)
	#train_data = data[:int(lds*0.8)]
	#test_data = data[int(lds*0.8):]
	
	#source_ids = load_vocabulary(args.SOURCE_VOCAB)
	#target_ids = load_vocabulary(args.TARGET_VOCAB)
	#maxlen_asm = max(map(lambda xy: len(xy[0]),data)) + 1
	#maxsize_c  = max(map(lambda xy: xy[1],data)) + 1
	#print(maxlen_asm,maxsize_c)
	
	n_maxsize = 800
	
	v_eos_src = len(asm_vocab)
	asm_vocab += ['__EOS__']
	src_vocab_len = len(asm_vocab)
 
	model = Seq2Tree_Flatten(n_layer, src_vocab_len, c_syntax_arr, n_unit, v_eos_src, n_maxsize)
	#serializers.load_npz('models/iter_1400_time_2019_01_22_09_37_45.npz',model)
	#serializers.load_npz('save_models/models_prestudy_edit_dist_0.65/iter_47600__edit_dist_0.691504__time_2018_12_28_01_36_30.npz',model)
	
	#print(train_data[0])
		
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
				sd = '__NORMALIZED_%d' % names.index(sd)
			ts.append(sd)
		return ts
	
	logt = 100
	
	def calcurate_edit_distance():
		ratio = 0.0
		ds = [data.get_test(-1) for _ in range(100)]
		for fr,to in ds:
			result = model.translate([fr])[0]
			result = tree2normalizedsentense(result)
			to = tree2normalizedsentense(to)
			#print(' '.join(to))
			#print(' '.join(result))
			#print('-'*100)
			ratio += 1.0 - edit_distance.SequenceMatcher(to,result).ratio()
		#print(ratio)
		ratio /= len(ds)
		return ratio
	
	#print('calc')
	#calcurate_edit_distance()
	
	train_iter_step = 100
	
	@chainer.training.make_extension()
	def translate(trainer):
		nonlocal logt
		logt += train_iter_step
		serializers.save_npz('models/iter_%s_time_%s.npz' % (logt,getstamp()),model)
		eds = calcurate_edit_distance()
		print('edit distance:',eds)
		with open('log/iter_%s_editdist_%f_time_%s.txt' % (logt,eds,getstamp()),'w') as fp:
			res = ""
			for _ in range(10):
				source,target = data.get_test(-1)
				result = model.translate([model.xp.array(source)])[0]
				print('got result')
				target = tree2normalizedsentense(target,normalize=False)
				result = tree2normalizedsentense(result,normalize=False)
				
				source_sentence = ' '.join([asm_vocab[x] for x in source])
				target_sentence = ' '.join(target)
				result_sentence = ' '.join(result)
				res += ('# source : ' + source_sentence) + "\n"
				res += ('# expect : ' + target_sentence) + "\n"
				#print('# result_array : ' + ' '.join(["%d" % y for y in result]))
				res += ('# result : ' + result_sentence) + "\n"
			fp.write(res)
		
		serializers.save_npz('models/iter_%s_editdist_%f_time_%s.npz' % (logt,eds,getstamp()),model)
		
		#import code
		#code.interact(local={'d':model.embed_x})
		#print('embed data : ', model.embed_x.W)

	return (model,data,translate)
		


def main():
	import sys
	sys.setrecursionlimit(10000)

	global train_iter_step 
	model,dataset,translate = init_seq2tree()
	#for _ in range(10):
	#	translate(None)
	#exit()
	
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)
	
	train_iter = chainer.iterators.SerialIterator(dataset, 64)
	updater = training.updaters.StandardUpdater(train_iter, optimizer,converter=convert)
	trainer = training.Trainer(updater, (10000000, 'epoch'), out='result')

	trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
	trainer.extend(extensions.PrintReport(
		['epoch', 'iteration', 'main/loss', 'validation/main/loss',
		 'main/perp', 'validation/main/perp', 'validation/main/bleu',
		 'elapsed_time']),
		trigger=(1, 'iteration'))
	
	trainer.extend(translate, trigger=(train_iter_step, 'iteration'))
	trainer.run()

main()

