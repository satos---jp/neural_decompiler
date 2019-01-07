#coding: utf-8

import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers





def preprocess_train_data(seq,tree):
	




def sequence_embed(embed, xs):
	if len(xs[0].shape) == 1: #通常の1次元配列のやつ
		x_len = [len(x) for x in xs]
		x_section = np.cumsum(x_len[:-1])
		ex = embed(F.concat(xs, axis=0))
		exs = F.split_axis(ex, x_section, 0)
		return exs
	else:
		ls = len(embed)
		ds = [sequence_embed(embed[i],[x[i] for x in xs]) for i in range(ls)]
		res = F.concat([y, eos_dst], axis=1)
		return res

class Seq2Tree(chainer.Chain):
	def __init__(self):
		super(Seq2Tree, self).__init__()
		
		# a :: 直前の挙動
		# p :: 親の挙動
		# n :: 現在のタイプ
		
		self.action_dim = 20
		self.parentact_dim = 20 
		self.nodetype_dim = 20
		
		self.eos_start
		self.eos_end

		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			self.embed_y_a = L.EmbedID(n_action_vocab, self.action_dim)
			self.embed_y_p = L.EmbedID(n_parentact_vocab, self.parentact_dim)
			self.embed_y_t = L.EmbedID(n_nodetype_vocab, self.nodetype_dim)
			
			self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
			self.W = L.Linear(n_units, n_target_vocab)
		
		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.v_eos_dst = v_eos_dst
		self.n_maxlen = n_maxlen
		

	def forward(self, xs,ys):
		# ys is concat of [a:p:n], which are determined before training.
		# とりあえずattentionは後でつけます...(とりあえず動かす)
		exs = sequence_embed(self.embed_x, xs)
		hx, cx, _ = self.encoder(None, None, exs)
		
		ys_as_input = [F.concat([self.eos_start, y], axis=0) for y in ys]
		eys = sequence_embed((self.embed_y_a,self.embed_y_p,self.embed_y_t), ys_as_input)
		_, _, os = self.decoder(hx, cx, eys)
		
		# from "Neural Machine Translation by Jointly Learning to Align and Translate" page 14.
		#attentions = F.softmax(F.tanh(self.Wa(y)+self.Ua(xs))
		
		ys_act = [F.concat([y[0], self.eos_end], axis=0) for y in ys]
		
		concat_os = F.concat(os, axis=0)
		concat_ys_out = F.concat(ys_out, axis=0)
		
		batch = len(xs)
		loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
		chainer.report({'loss': loss}, self)
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
