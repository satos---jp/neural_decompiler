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


def convert(batch, device):
	def to_device_batch(batch):
		return batch

	return {'xs': to_device_batch([x for x, _ in batch]),
			'ys': to_device_batch([y for _, y in batch])}

import datetime
def getstamp():
	return datetime.datetime.today().strftime("%Y_%m_%d_%H_%M_%S")

import code
import edit_distance
import random
import pickle
import csrc2ast



def init_seq2seq():
	n_layer = 4
	n_unit = 128
	
	n_maxlen = 100
	
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



def init_seq2tree():
	n_layer = 4
	n_unit = 128
	
	n_maxdepth = 10
	
	with open('data_asm_ast.pickle','rb') as fp:
			(data,asm_vocab,c_syntax_arr) = pickle.load(fp)
	
	#print(data[0])
	#data.data = list(filter(lambda xy: len(xy[0])<20 and len(xy[1])<10,data.data))
	#data = list(filter(lambda xy: len(xy[1])<n_maxsize,data))
	data = data[:1000000]
	print(len(data))
	
	lds = len(data)
	#train_data = data[:int(lds*0.8)]
	#test_data = data[int(lds*0.8):]
	train_data = data
	test_data = data
	#TODO(satos) sugu modosukoto.
	
	#source_ids = load_vocabulary(args.SOURCE_VOCAB)
	#target_ids = load_vocabulary(args.TARGET_VOCAB)
	maxlen_asm = max(map(lambda xy: len(xy[0]),data)) + 1
	maxsize_c  = max(map(lambda xy: xy[1],data)) + 1
	print(maxlen_asm,maxsize_c)
	
	v_eos_src = len(asm_vocab)
	asm_vocab += ['__EOS__']
	src_vocab_len = len(asm_vocab)
 
	model = Seq2Tree(n_layer, src_vocab_len, c_syntax_arr, n_unit, v_eos_src, n_maxdepth)
	#serializers.load_npz('models/iter_5600_editdist_0.866712_time_2019_01_15_04_50_45.npz',model)
	#serializers.load_npz('save_models/models_prestudy_edit_dist_0.65/iter_47600__edit_dist_0.691504__time_2018_12_28_01_36_30.npz',model)
	
	#print(train_data[0])
	ddata  = list(map(lambda xy: np.array(xy[0],np.int32),train_data))
	dlabel = list(map(lambda xy: xy[2],train_data))
	dataset = list(zip(ddata,dlabel))
	
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
	
	logt = 0
	
	def calcurate_edit_distance():
		ratio = 0.0
		ds = random.sample(test_data, 100)
		for fr,_,to in ds:
			result = model.translate([model.xp.array(fr)])[0]
			result = tree2normalizedsentense(result)
			to = tree2normalizedsentense(to)
			#print(to,result)
			ratio += 1.0 - edit_distance.SequenceMatcher(to,result).ratio()
		#print(ratio)
		ratio /= len(ds)
		return ratio
	
	train_iter_step = 100
	
	@chainer.training.make_extension()
	def translate(trainer):
		eds = calcurate_edit_distance()
		print('edit distance:',eds)
		nonlocal logt
		logt += train_iter_step
		with open('log/iter_%s_editdist_%f_time_%s.txt' % (logt,eds,getstamp()),'w') as fp:
			res = ""
			for _ in range(10):
				source,_, target = test_data[np.random.choice(len(test_data))]
				result = model.translate([model.xp.array(source)])[0]
				
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

	return (model,dataset,translate)
		


def main():
	model,dataset,translate = init_seq2tree()
	#for _ in range(10):
	#	translate(None)
	#exit()
	
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)
	
	train_iter = chainer.iterators.SerialIterator(dataset, 64)
	updater = training.updaters.StandardUpdater(train_iter, optimizer,converter=convert)
	trainer = training.Trainer(updater, (100000000, 'epoch'), out='result')

	trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
	trainer.extend(extensions.PrintReport(
		['epoch', 'iteration', 'main/loss', 'validation/main/loss',
		 'main/perp', 'validation/main/perp', 'validation/main/bleu',
		 'elapsed_time']),
		trigger=(1, 'iteration'))
	
	train_iter_step = 100
	trainer.extend(translate, trigger=(train_iter_step, 'iteration'))
	trainer.run()

main()

