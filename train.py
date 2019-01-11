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



def main():
	n_layer = 4
	n_unit = 128
	
	n_maxlen = 100
	
	with open('data.pickle','rb') as fp:
	#with open('data_simple.pickle','rb') as fp:
			(data,c_vocab,asm_vocab) = pickle.load(fp)
	
	#data.data = list(filter(lambda xy: len(xy[0])<20 and len(xy[1])<10,data.data))
	data = list(filter(lambda xy: len(xy[1])<n_maxlen,data))
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
 
	model = Seq2seq_with_Att(n_layer, src_vocab_len, dst_vocab_len , n_unit, v_eos_src, v_eos_dst, n_maxlen)
	#serializers.load_npz('models/iter_100__time_2019_01_10_02_01_29.npz',model)
	#serializers.load_npz('save_models/models_prestudy_edit_dist_0.65/iter_47600__edit_dist_0.691504__time_2018_12_28_01_36_30.npz',model)


	def normalize_sentense(s):
		ts = []
		names = []
		for d in s:
			sd = c_vocab[d]
			if sd[:5]=='__ID_':
				if not sd in names:
					names.append(sd)
				d = -1-names.index(sd)
			ts.append(d)
		return ts
	
	"""
	def calc_dist_dist(ds):
		res = []
		i = 0
		for fr,to in ds:
			i += 1
			if i % 100 == 0:
				print('cnt',i)
			result = model.translate([model.xp.array(fr)])[0]
			result = list(result)
			if len(to)>0:
				source_sentence = ' '.join(['%02x' % x for x in fr])
				target_sentence = ' '.join([c_vocab[y] for y in to])
				result_sentence = ' '.join([c_vocab[y] for y in result])	
				print('=========================================')
				print(len(to),ratio)
				print(source_sentence)			
				print(target_sentence)			
				print(result_sentence)	
			
			to = normalize_sentense(to)
			result = normalize_sentense(result)
			ratio = 1.0 - edit_distance.SequenceMatcher(to,result).ratio()
			print(to,result)
		
			res.append((len(to),ratio))
		return res
	
	ds = random.sample(test_data, 10000)
	#ds = test_data
	print('start calc',len(ds))
	tds = calc_dist_dist(ds)
	print('calced',len(tds))
	#with open('dist_dist.pickle','wb') as fp:
	#		pickle.dump(tds,fp)
	exit()
	"""
			
		
	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)
	
	
	
	#ddata  = list(map(lambda xy: np.array(xy[0] + (maxlen_asm - len(xy[0])) * [v_eos],np.int32),data.data))
	#dlabel = list(map(lambda xy: np.array(xy[1] + (maxlen_src - len(xy[1])) * [v_eos],np.int32),data.data))
	ddata  = list(map(lambda xy: np.array(xy[0] + [v_eos_src],np.int32),train_data))
	dlabel = list(map(lambda xy: np.array(xy[1] + [v_eos_dst],np.int32),train_data))
	#dataset = chainer.datasets.tuple_dataset.TupleDataset(ddata,dlabel)
	dataset = list(zip(ddata,dlabel))
	#code.interact(local={'data': dataset})
	
	train_iter = chainer.iterators.SerialIterator(dataset, 64)
	updater = training.updaters.StandardUpdater(train_iter, optimizer,converter=convert)
	trainer = training.Trainer(updater, (100000000, 'epoch'), out='result')

	trainer.extend(extensions.LogReport(trigger=(1, 'iteration')))
	trainer.extend(extensions.PrintReport(
		['epoch', 'iteration', 'main/loss', 'validation/main/loss',
		 'main/perp', 'validation/main/perp', 'validation/main/bleu',
		 'elapsed_time']),
		trigger=(1, 'iteration'))

	logt = 100
	train_iter_step = 1
	
	

	
	
	def calcurate_edit_distance():
		ratio = 0.0
		ds = random.sample(test_data, 100)
		for fr,to in ds:
			result = model.translate([model.xp.array(fr)])[0]
			result = list(result)
			to = normalize_sentense(to)
			result = normalize_sentense(result)
			#print(to,result)
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
		
				source_sentence = ' '.join([asm_vocab[x] for x in source])
				target_sentence = ' '.join([c_vocab[y] for y in target])
				result_sentence = ' '.join([c_vocab[y] for y in result])
				res += ('# source : ' + source_sentence) + "\n"
				res += ('# expect : ' + target_sentence) + "\n"
				#print('# result_array : ' + ' '.join(["%d" % y for y in result]))
				res += ('# result : ' + result_sentence) + "\n"
			fp.write(res)
		
		serializers.save_npz('models/iter_%s_editdist_%f_time_%s.npz' % (logt,eds,getstamp()),model)
		
		#import code
		#code.interact(local={'d':model.embed_x})
		#print('embed data : ', model.embed_x.W)

	trainer.extend(translate, trigger=(train_iter_step, 'iteration'))

	trainer.run()

main()

