import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers
import c_cfg
	
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
	def __init__(self, n_layers, n_source_vocab, trans_data, n_units,v_eos_src,n_maxdepth):
		super(Seq2Tree, self).__init__()

		# for each nodetype, for each move, the result array.
		self.trans_data = trans_data
		self.embed_idx = []
		ns = 0
		def inc():
			nonlocal ns
			ns += 1
			return ns-1
		self.embed_idx = [[[inc() for v in vs] for vs in moves] for moves in self.trans_data]
		self.embed_root_idx = ns
		self.embed_y_size = ns+1
		
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			self.embed_y = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
			self.W = [None for _ in self.trans_data]
			for i,moves in enumerate(self.trans_data):
				if len(moves)>1:
					self.W[i] = L.Linear(n_units, len(moves))
			
			self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units*3, n_units, 0.1)
			#self.W = L.Linear(n_units, n_target_vocab)
			
			self.att = Attention(n_units)
		
		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.n_maxdepth = n_maxdepth
		self.v_eos_dst = c_cfg.rootidx
	
	def forward(self, xs, ys):
		#print(ys)
		#exit()
		#xs = [x[::-1] for x in xs]
		
		#eos_dst = self.xp.array([self.v_eos_dst], np.int32)
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
		
		concat_oss = [[] for _ in self.trans_data]
		concat_ys_outs = [[] for _ in self.trans_data]
		def rec_LSTM(node,eidx,nhxncx):
			(nhx,ncx) = nhxncx
			ntype,nchoice,children = node
			#eidx = self.embed_idx[ntype][ppos]
			ev = self.embed_y(np.array([eidx]))[0]
			ctx = self.att(nhx)
			tv = F.reshape(F.concat([ctx,ev],axis=0),(1,self.n_units * 3))
			thx,tcx,nos = self.decoder(nhx,ncx,[tv])
			nos = nos[0]
			#wnos = self.W[ntype](nos)
			
			if len(self.trans_data[ntype])>1:
				concat_oss[ntype].append(nos)
				concat_ys_outs[ntype].append(nchoice)
			# otherwise, we don't have to train.
			for i,ch in enumerate(children):
				#print(ntype,nchoice,i)
				teidx = self.embed_idx[ntype][nchoice][i]
				rec_LSTM(ch,teidx,(thx,tcx))
		
		for i,(y,hxs) in enumerate(zip(ys,xs_states)):
			nhx,ncx = hx[i],cx[i]
			ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
			nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
			
			assert y[0] == self.v_eos_dst
			self.att.init_hxs(hxs)
			ridx = self.embed_root_idx
			rec_LSTM(y,ridx,(nhx,ncx))
			self.att.reset()
		
		#loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
		
		loss_s = []
		for i,(concat_os,concat_ys_out) in enumerate(zip(concat_oss,concat_ys_outs)):
			assert len(concat_os) == len(concat_ys_out)
			if len(concat_os)==0:
				continue
			#print(concat_os)
			#print(concat_ys_out)
			d = F.softmax_cross_entropy(self.W[i](F.concat(concat_os,axis=0)), np.array(concat_ys_out), reduce='no')
			loss_s.append(d)
		
		#print(loss_s)
		
		loss = F.sum(F.concat(loss_s,axis=0))/ batch
		
		chainer.report({'loss': loss}, self)
		#n_words = concat_ys_out.shape[0]
		#perp = self.xp.exp(loss.array * batch / n_words)
		#chainer.report({'perp': perp}, self)
		return loss
	
	def translate(self, xs):
		EOS_DST = self.v_eos_dst
		batch = len(xs)
		
		#beam_with = 3
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			
			exs = sequence_embed(self.embed_x, xs)
			hx, cx, xs_states = self.encoder(None, None, exs)
			#print(hx.shape,cx.shape,(1,xs_states[0].shape))
			#sprint(xs_states)
			hx = F.transpose(hx,axes=(1,0,2))
			cx = F.transpose(cx,axes=(1,0,2))
			
			ivs = [self.embed_y(self.xp.full(1,i,np.int32))[0] for i in range(self.embed_y_size)]
			v = ivs[EOS_DST]
			
			result = []



			def expand_tree(ntype,eidx,nhxncx,ndepth=0):
				(nhx,ncx) = nhxncx
				if ndepth > self.n_maxdepth:
					return (ntype,-1,[])
				#eidx = self.embed_idx[ntype][ppos]
				ev = self.embed_y(np.array([eidx]))[0]
				ctx = self.att(nhx)
				tv = F.reshape(F.concat([ctx,ev],axis=0),(1,self.n_units * 3))
				thx,tcx,nos = self.decoder(nhx,ncx,[tv])
				nos = nos[0]
				
				if len(self.trans_data[ntype])>1:
					wy = self.W[ntype](nos)
					#wy = F.reshape(F.log_softmax(F.reshape(wy,(1,self.vs_target_vocab[ntype])),axis=1),(self.vs_target_vocab[ntype],)).data
					nchoice = F.argmax(wy.data).data.astype(np.int32).item()
				else:
					nchoice = 0
				
				#print(ntype,nchoice)
				#print(c_cfg.idx2nodetype(ntype))
				ctypes = self.trans_data[ntype][nchoice]
				resv = []
				for i,ct in enumerate(ctypes):
					teidx = self.embed_idx[ntype][nchoice][i]
					resv.append(expand_tree(ct,teidx,(thx,tcx),ndepth+1))
				return (ntype,nchoice,resv)
			
			for i,hxs in enumerate(xs_states):
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
		
				self.att.init_hxs(hxs)
				ridx = self.embed_root_idx
				
				# TODO(satos) What is the beam search for tree!?
				# now, beam search is ommited.
				tree = expand_tree(self.v_eos_dst,ridx,(nhx,ncx))
				self.att.reset()
				result.append(tree)
			
		return result
	
	
