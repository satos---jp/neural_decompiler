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
	
class Seq2Tree(chainer.Chain):
	def __init__(self, n_layers, n_source_vocab, children_types, n_units,v_eos_src,v_eos_dst,n_maxlen):
		super(Seq2Tree, self).__init__()

		self.children_types = children_types
		self.vs_target_vocab = [len(d) for d in self.children_types]
		self.embed_idx = []
		ns = 0
		for d in self.vs_target_vocab:
			self.embed_idx.append([i+ns for i in range(d)])
			ns += d
		self.embed_y_size = ns
		
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			self.embed_y = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
			for i,lv in enumerate(self.vs_target_vocab):
				self.W[i] = L.Linear(n_units, lv)
			
			self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units*3, n_units, 0.1)
			#self.W = L.Linear(n_units, n_target_vocab)
			
			self.att = Attention(n_units)
		
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
		#ys_in = [F.concat([eos_dst, y], axis=0) for y in ys]
		#ys_out = [F.concat([y, eos_dst], axis=0) for y in ys]

		# Both xs and ys_in are lists of arrays.
		exs = sequence_embed(self.embed_x, xs)
		#eys = sequence_embed(self.embed_y, ys_in)
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
		
		
		concat_oss = []
		concat_ys_outs = []
		def rec_LSTM(node,ptype,ppos,(nhx,ncx)):
			ntype,nchoice,children = node
			eidx = self.embed_idx[ptype][ppos]
			ev = self.embed_y(np.array([eidx]))
			ctx = self.att(nhx)
			tv = F.reshape(F.concat([ctx,ev],axis=0),(1,self.n_units * 3))
			thx,tcx,nos = self.decoder(nhx,ncx,[tv])
			#wnos = self.W[ntype](nos)
			
			concat_oss[ntype].append(nos)
			concat_ys_outs[ntype].append(nchoice)
			for i,ch in enumerate(children):
				rec_LSTM(ch,ntype,i,(thx,tcx))
		
		
		for i,(y,hxs) in enumerate(zip(eys,xs_states)):
			nhx,ncx = hx[i],cx[i]
			ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
			nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
			
			assert y[0] == self.v_eos_dst
			self.att.init_hxs(hxs)
			rec_LSTM(y,0,0,(nhx,ncx))
			self.att.reset()
		
		#loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
		loss_s = [
			F.softmax_cross_entropy(self.W[i](np.array(concat_os)), np.array(concat_ys_out), reduce='no')
			for i,(concat_os,concat_ys_out) in enumerate(zip(concat_oss,concat_ys_outs))
		]
		
		loss = F.sum(F.concat(loss_s))/ batch
		
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
			
			def expand_tree(ntype,ptype,ppos,(nhx,ncx)):
				eidx = self.embed_idx[ptype][ppos]
				ev = self.embed_y(np.array([eidx]))
				ctx = self.att(nhx)
				tv = F.reshape(F.concat([ctx,ev],axis=0),(1,self.n_units * 3))
				thx,tcx,nos = self.decoder(nhx,ncx,[tv])
				wy = self.W[ntype](nos)
				
				#wy = F.reshape(F.log_softmax(F.reshape(wy,(1,self.vs_target_vocab[ntype])),axis=1),(self.vs_target_vocab[ntype],)).data
				nchoice = F.argmax(wy.data).astype(np.int32)
				
				ctypes = self.children_types[ntype][nchoice]
				resv = []
				for i,ct in enumerate(ctypes):
					res.append(expand_tree(ct,ntype,i,(thx,tcx)))
				return (ntype,nchoice,resv)
			
			for i,hxs in enumerate(xs_states):
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
				self.att.init_hxs(hxs)
				
				# TODO(satos) What is the beam search for tree!?
				# now, beam search is ommited.
				
				tree = expand_tree(self.v_eos_dst,0,0,(nhx,ncx))
				self.att.reset()
				result.append(tree)
			
		return result
	
	
