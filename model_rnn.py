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
		self.n_target_vocab = n_target_vocab

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
		#print(hx.shape,cx.shape)
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
		"""
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
		"""
		
		beam_with = 3
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			xs = [x[::-1] for x in xs]
			exs = sequence_embed(self.embed_x, xs)
			hx, cx, _ = self.encoder(None, None, exs)
			#print(hx.shape,cx.shape,(1,xs_states[0].shape))
			#sprint(xs_states)
			hx = F.transpose(hx,axes=(1,0,2))
			cx = F.transpose(cx,axes=(1,0,2))
			
			ivs = [self.embed_y(self.xp.full(1,i,np.int32))[0] for i in range(self.n_target_vocab)]
			v = ivs[EOS_DST]
			
			result = []
			
			for i in range(len(xs)):
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
				beam_data = [(0.0,([],v,nhx,ncx))]
				
				for j in range(self.n_maxlen):
					to_beam = []
					for r,(kd,v,nhx,ncx) in beam_data:
						tv = F.reshape(v,(1,self.n_units))
						thx,tcx,ys = self.decoder(nhx,ncx,[tv])
						
						wy = self.W(ys[0]).data[0]
						#print(wy.shape,wy)
						wy = F.reshape(F.log_softmax(F.reshape(wy,(1,self.n_target_vocab)),axis=1),(self.n_target_vocab,)).data
						#print(wy.shape)
						to_beam += [(r+nr,(kd + [i],ivs[i],thx,tcx)) for i,nr in enumerate(wy)]
					
					#print(to_beam[0][0])
					#print(list(map(lambda a: a[0],to_beam[:10])))
					beam_data = sorted(to_beam)[::-1][:beam_with]
					#print(list(map(lambda a: a[0],beam_data[:10])))
					
				result.append(beam_data[0][1][0])
		# Remove EOS taggs
		outs = []
		for y in result:
			if EOS_DST in y:
				y = y[:y.index(EOS_DST)]
			#print(y)
			outs.append(y)
		return outs

