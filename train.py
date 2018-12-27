#! /usr/bin/python3
#coding: utf-8

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers

def sequence_embed(embed, xs):
	x_len = [len(x) for x in xs]
	x_section = np.cumsum(x_len[:-1])
	ex = embed(F.concat(xs, axis=0))
	exs = F.split_axis(ex, x_section, 0)
	return exs

class Seq2seq(chainer.Chain):
	def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,v_eos_src,v_eos_dst,n_maxlen):
		super(Seq2seq, self).__init__()
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			self.embed_y = L.EmbedID(n_target_vocab, n_units)
			self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
			self.W = L.Linear(n_units, n_target_vocab)

		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.v_eos_dst = v_eos_dst
		self.n_maxlen = n_maxlen

	def forward(self, xs, ys):
		#print(xs,ys)
		#exit()
		xs = [x[::-1] for x in xs]
		
		eos_dst = self.xp.array([self.v_eos_dst], np.int32)
		ys_in = [F.concat([eos_dst, y], axis=0) for y in ys]
		ys_out = [F.concat([y, eos_dst], axis=0) for y in ys]

		# Both xs and ys_in are lists of arrays.
		exs = sequence_embed(self.embed_x, xs)
		eys = sequence_embed(self.embed_y, ys_in)
		#print(list(map(lambda x: len(x),exs)))
		#print(list(map(lambda x: len(x),eys)))
		#exit()
		
		batch = len(xs)
		# None represents a zero vector in an encoder.
		hx, cx, _ = self.encoder(None, None, exs)
		_, _, os = self.decoder(hx, cx, eys)

		#print(list(map(lambda x: len(x),os)))
		
		# It is faster to concatenate data before calculating loss
		# because only one matrix multiplication is called.
		concat_os = F.concat(os, axis=0)
		concat_ys_out = F.concat(ys_out, axis=0)
		loss = F.sum(F.softmax_cross_entropy(
			self.W(concat_os), concat_ys_out, reduce='no')) / batch

		chainer.report({'loss': loss}, self)
		n_words = concat_ys_out.shape[0]
		perp = self.xp.exp(loss.array * batch / n_words)
		chainer.report({'perp': perp}, self)
		return loss


	def translate(self, xs):
		EOS_DST = self.v_eos_dst
		batch = len(xs)
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			xs = [x[::-1] for x in xs]
			exs = sequence_embed(self.embed_x, xs)
			h, c, _ = self.encoder(None, None, exs)
			ys = self.xp.full(batch, EOS_DST, np.int32)
			result = []
			for i in range(self.n_maxlen):
				eys = self.embed_y(ys)
				eys = F.split_axis(eys, batch, 0)
				h, c, ys = self.decoder(h, c, eys)
				cys = F.concat(ys, axis=0)
				wy = self.W(cys)
				ys = self.xp.argmax(wy.array, axis=1).astype(np.int32)
				result.append(ys)

		# Using `xp.concatenate(...)` instead of `xp.stack(result)` here to
		# support NumPy 1.9.
		result = self.xp.concatenate([self.xp.expand_dims(x, 0) for x in result]).T

		# Remove EOS taggs
		outs = []
		for y in result:
			inds = np.argwhere(y == EOS_DST)
			if len(inds) > 0:
				y = y[:inds[0, 0]]
			outs.append(y)
		return outs

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
			(data,c_vocab) = pickle.load(fp)
	
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
	
	v_eos_src = 256 #現状asmなので
	src_vocab_len = v_eos_src + 1
	
	v_eos_dst = len(c_vocab)
	c_vocab += ['__EOS__']
	dst_vocab_len = len(c_vocab)
 
	model = Seq2seq(n_layer, src_vocab_len, dst_vocab_len , n_unit, v_eos_src, v_eos_dst, n_maxlen)
	serializers.load_npz('models/iter_24100__edit_dist_0.660912__time_2018_12_25_18_46_26.npz',model)

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

	logt = 24100
	train_iter_step = 100
	
	def calcurate_edit_distance():
		ratio = 0.0
		ds = random.sample(test_data, 100)
		for fr,to in ds:
			result = model.translate([model.xp.array(fr)])[0]
			result = list(result)
			#print(to,result)
			ratio += 1.0 - edit_distance.SequenceMatcher(to,result).ratio()
		#print(ratio)
		ratio /= len(ds)
		return ratio
	
	@chainer.training.make_extension()
	def translate(trainer):
		eds = calcurate_edit_distance()
		print('edit distance:',calcurate_edit_distance())
		nonlocal logt
		logt += train_iter_step
		with open('log/iter_%s__time_%s.txt' % (logt,getstamp()),'w') as fp:
			res = ""
			for _ in range(10):
				source, target = test_data[np.random.choice(len(test_data))]
				result = model.translate([model.xp.array(source)])[0]
		
				source_sentence = ' '.join(['%02x' % x for x in source])
				target_sentence = ' '.join([c_vocab[y] for y in target])
				result_sentence = ' '.join([c_vocab[y] for y in result])
				res += ('# source : ' + source_sentence) + "\n"
				res += ('# expect : ' + target_sentence) + "\n"
				#print('# result_array : ' + ' '.join(["%d" % y for y in result]))
				res += ('# result : ' + result_sentence) + "\n"
			fp.write(res)
		
		serializers.save_npz('models/iter_%s__edit_dist_%f__time_%s.npz' % (logt,eds,getstamp()),model)
		
		#import code
		#code.interact(local={'d':model.embed_x})
		#print('embed data : ', model.embed_x.W)

	trainer.extend(translate, trigger=(train_iter_step, 'iteration'))

	trainer.run()

main()

