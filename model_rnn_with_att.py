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


# from https://arxiv.org/pdf/1409.0473.pdf. Global attention.
class Attention(chainer.Chain):
	def __init__(self, n_units):
		super(Attention, self).__init__()
		with self.init_scope():
			self.att_W = L.Linear(n_units, n_units)
			self.att_U = L.Linear(n_units*2, n_units)
			self.att_v = L.Linear(n_units, 1)
		
		self.n_units = n_units
		self.Uhxs = None
	
	def init_hxs(self,hxs):
		self.hxs = hxs
		self.Uhxs = self.att_U(hxs)
		self.Uhxs_len = hxs.shape[0]
	
	def forward(self,nhx):
		nhx_for_att = nhx[-1]
		Wnhx = self.att_W(nhx_for_att)
		#print(Wnhx.shape,Uhxs.shape)
		#print((Wnhx + Uhxs).shape)
		
		att_rates = F.softmax(F.reshape(self.att_v(
				F.tanh(
					Wnhx + self.Uhxs
				)
		),(1,self.Uhxs_len)))
		#print(hxs.shape,att_rates.shape)
		#print(att_rates)
		#print(F.matmul(att_rates,hxs).shape)
		ctx = F.matmul(att_rates,self.hxs)[0]
		#print(ctx.shape)
		return ctx
		
	def reset(self):
		self.hxs = None
		self.Uhxs = None
	
class Seq2seq_with_Att(chainer.Chain):
	def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,v_eos_src,v_eos_dst,n_maxlen):
		super(Seq2seq_with_Att, self).__init__()
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			self.embed_y = L.EmbedID(n_target_vocab, n_units)
			self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units*3, n_units, 0.1)
			self.W = L.Linear(n_units, n_target_vocab)
			
			self.att = Attention(n_units)
		
		self.n_target_vocab = n_target_vocab
		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.v_eos_dst = v_eos_dst
		self.n_maxlen = n_maxlen

	def forward(self, xs, ys):
		#print(xs,ys)
		#exit()
		#xs = [x[::-1] for x in xs]
		
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
		hx, cx, xs_states = self.encoder(None, None, exs)
		#_, _, os = self.decoder(hx, cx, eys)
		
		hx = F.transpose(hx,axes=(1,0,2))
		cx = F.transpose(cx,axes=(1,0,2))
		
		#print(batch,hx.shape,cx.shape,(batch,xs_states[0].shape))
		os = []
		for i,(y,hxs) in enumerate(zip(eys,xs_states)):
			nhx,ncx = hx[i],cx[i]
			ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
			nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
			
			self.att.init_hxs(hxs)
			
			nos = []
			for v in y:
				ctx = self.att(nhx)
				tv = F.reshape(F.concat([ctx,v],axis=0),(1,self.n_units * 3))
				#print(tv.shape)
				#print(nhx.shape,ncx.shape,tv.shape)
				thxs,tcxs,_ = self.decoder(nhx,ncx,[tv])
				
				nhx = thxs
				ncx = tcxs
				#print(nhx[-1][0].dtype)
				nos.append(nhx[-1][0])
			
			nos = F.stack(nos)
			#print(nos.shape)
			os.append(nos)
			
			self.att.reset()
		
		#print(list(map(lambda x: len(x),os)))
		
		# It is faster to concatenate data before calculating loss
		# because only one matrix multiplication is called.
		concat_os = F.concat(os, axis=0)
		#print(concat_os.dtype)
		concat_ys_out = F.concat(ys_out, axis=0)
		#print(concat_ys_out.dtype)
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
		
		beam_with = 3
		with chainer.no_backprop_mode(), chainer.using_config('train', False):

			exs = sequence_embed(self.embed_x, xs)
			hx, cx, xs_states = self.encoder(None, None, exs)
			#print(hx.shape,cx.shape,(1,xs_states[0].shape))
			#sprint(xs_states)
			hx = F.transpose(hx,axes=(1,0,2))
			cx = F.transpose(cx,axes=(1,0,2))
			
			ivs = [self.embed_y(self.xp.full(1,i,np.int32))[0] for i in range(self.n_target_vocab)]
			v = ivs[EOS_DST]
			
			result = []
			
			for i,hxs in enumerate(xs_states):
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
				self.att.init_hxs(hxs)
				beam_data = [(0.0,([],v,nhx,ncx))]
				
				for j in range(self.n_maxlen):
					to_beam = []
					for r,(kd,v,nhx,ncx) in beam_data:
						ctx = self.att(nhx)
						#print(ctx.shape,v.shape)
						tv = F.reshape(F.concat([ctx,v],axis=0),(1,self.n_units * 3))
						#print(tv.shape)
						#print(nhx.shape,ncx.shape,tv.shape)
						thx,tcx,ys = self.decoder(nhx,ncx,[tv])
						
						wy = self.W(ys[0]).data[0]
						wy = F.reshape(F.log_softmax(F.reshape(wy,(1,self.n_target_vocab)),axis=1),(self.n_target_vocab,)).data
						#print(wy.shape)
						to_beam += [(r+nr,(kd + [i],ivs[i],thx,tcx)) for i,nr in enumerate(wy)]
					
					#print(to_beam[0][0])
					#print(list(map(lambda a: a[0],to_beam)))
					beam_data = sorted(to_beam)[::-1][:beam_with]
					#print(list(map(lambda a: a[0],beam_data)))
				
				self.att.reset()
				result.append(beam_data[0][1][0])			
		
		# Remove EOS taggs
		outs = []
		for y in result:
			if EOS_DST in y:
				y = y[:y.index(EOS_DST)]
			#print(y)
			outs.append(y)
		return outs
	
	
