import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import serializers

from chainer import cuda
from cuda_setting import isgpu
	
if isgpu:
	xp = cuda.cupy
else:
	xp = np

def sequence_embed(embed, xs):
	x_len = [len(x) for x in xs]
	x_section = np.cumsum(x_len[:-1])
	ex = embed(F.concat(xs, axis=0))
	exs = F.split_axis(ex, x_section, 0)
	return exs


from model_attention import GlobalGeneralAttention
import random


class Seq2Tree(chainer.Chain):
	def __init__(self, n_layers, n_source_vocab, trans_data, n_units,v_eos_src,n_maxsize):
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
		
		self.choicerange = []
		self.is_trivial = []
		self.choice_idx = []
		s = 0
		for d in self.trans_data:
			ist = len(d)<=1
			self.is_trivial.append(ist)
			if ist:
				self.choicerange.append(None)
				self.choice_idx.append([0])
				continue
			b = s
			s += len(d)
			self.choicerange.append((b,s))
			self.choice_idx.append(list(range(b,s)))
		#self.choice_num_sum = sum(list(map(lambda d: len(d),self.trans_data)))
		self.n_choicables = s
		self.type_size = len(self.embed_idx)
		
		with self.init_scope():
			self.embed_x = L.EmbedID(n_source_vocab, n_units)
			self.embed_y = L.EmbedID(self.embed_y_size, n_units) # maybe mergable
			
			self.encoder = L.NStepBiLSTM(n_layers, n_units, n_units, 0.1)
			self.decoder = L.NStepLSTM(n_layers, n_units, n_units*2, 0.1)
			self.Wc = L.Linear(n_units*4, n_units)
			self.Ws = L.Linear(n_units, self.n_choicables)
			
			#self.att = Attention(n_units)
			self.att = GlobalGeneralAttention(n_units)
		
		self.n_layers = n_layers
		self.n_units = n_units
		self.v_eos_src = v_eos_src
		self.n_maxsize = n_maxsize
		self.rootidx = len(trans_data)-1
	
		
	def forward(self, xs, ys):
		batch = len(xs)
		xs = [xp.array(x[::-1]) for x in xs]
		exs = sequence_embed(self.embed_x, xs)
		
		def sample_path(y):
			my = y
			eidx = self.embed_root_idx
			ist = False
			res = []
			while True:
				ty,ch,cs = y
				ci = self.choice_idx[ty][ch]
				res.append((eidx,ci,self.is_trivial[ty]))
				lcs = len(cs)
				if lcs == 0:
					break
				i = random.randint(0,lcs-1)
				eidx = self.embed_idx[ty][ch][i]
				y = cs[i]
			#print(res)
			return res
		
		ys = [sample_path(y) for y in ys]
		ys_out = [xp.array(list(map(lambda a: a[1],d))) for d in ys]
		#print('ys out')
		#print(ys_out)
		ys_conds = [xp.array(list(map(lambda a: a[2],d)),dtype=xp.bool) for d in ys]
		#print(self.embed_y_size,self.n_all_choice)
		eys = sequence_embed(self.embed_y, [xp.array(list(map(lambda a: a[0],d))) for d in ys])
		
		hx, cx, xs_states = self.encoder(None, None, exs)
		hx = F.transpose(F.reshape(F.transpose(hx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
		cx = F.transpose(F.reshape(F.transpose(cx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
		
		_, _, os = self.decoder(hx, cx, eys)
		#print('decode')
		
		ctxs = [self.att(xh,yh) for (xh,yh) in zip(xs_states,os)]
		#print('attentioned')
		att_os = [F.tanh(self.Wc(F.concat([ch,yh],axis=1))) for (ch,yh) in zip(ctxs,os)]
		
		concat_os = F.concat(att_os, axis=0)
		concat_ys_out = F.concat(ys_out, axis=0)
		concat_cond = F.concat(ys_conds, axis=0)
		
		#print(concat_ys_out,concat_cond)
		sxe = F.softmax_cross_entropy(self.Ws(concat_os), concat_ys_out, reduce='no')
		sxec = F.where(concat_cond,xp.zeros(sxe.shape,dtype=xp.float32),sxe)
		#print(sxec)
		loss = F.sum(sxec) / batch
		#print('lossed')
		chainer.report({'loss': loss}, self)
		#exit()
		return loss
	
	def translate(self, xs):
		batch = len(xs)
		
		#beam_with = 3
		with chainer.no_backprop_mode(), chainer.using_config('train', False):
			xs = [xp.array(x[::-1]) for x in xs]
			exs = sequence_embed(self.embed_x, xs)
			hx, cx, xs_outputs = self.encoder(None, None, exs)
			#print(hx.shape,cx.shape,(1,xs_states[0].shape))
			#sprint(xs_states)
			hx = F.transpose(F.reshape(F.transpose(hx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
			cx = F.transpose(F.reshape(F.transpose(cx,(1,0,2)),(batch,self.n_layers,self.n_units*2)),(1,0,2))
			hx = F.transpose(hx,axes=(1,0,2))
			cx = F.transpose(cx,axes=(1,0,2))
			
			ivs = sequence_embed(self.embed_y,list(map(lambda i: xp.array([i]),range(self.embed_y_size))))
			v = ivs[self.embed_root_idx]
			
			result = []
			
			nsize = None
			
			for i in range(len(xs_outputs)):
				def expand_tree(ntype,eidx,nhxncx):
					nonlocal nsize
					(nhx,ncx) = nhxncx
					if nsize > self.n_maxsize:
						return (ntype,-1,[])
					nsize += 1
					#eidx = self.embed_idx[ntype][ppos]
					ev = ivs[eidx]
					thx,tcx,ys = self.decoder(nhx,ncx,[ev])
					yh = ys[0]
					ctx = self.att(xs_outputs[i],yh)
					att_yh = F.tanh(self.Wc(F.concat([ctx,yh],axis=1)))
				
					if self.is_trivial[ntype]:
						nchoice = 0
					else:
						choice_from,choice_to = self.choicerange[ntype]
						cl = choice_to - choice_from
						wy = self.Ws(att_yh).data[0][choice_from:choice_to]
						#print(wy.shape,wy)
						wy = F.reshape(F.log_softmax(F.reshape(wy,(1,cl))),(cl,))
						#wy = F.reshape(F.log_softmax(F.reshape(wy,(1,self.vs_target_vocab[ntype])),axis=1),(self.vs_target_vocab[ntype],)).data
						nchoice = F.argmax(wy.data).data.astype(np.int32).item()
				
					#print(ntype,nchoice)
					#print(c_cfg.idx2nodetype(ntype))
					ctypes = self.trans_data[ntype][nchoice]
					resv = []
					for j,ct in enumerate(ctypes):
						teidx = self.embed_idx[ntype][nchoice][j]
						resv.append(expand_tree(ct,teidx,(thx,tcx)))
					return (ntype,nchoice,resv)
				
				
				nhx,ncx = hx[i],cx[i]
				ncx = F.reshape(ncx,(ncx.shape[0],1,ncx.shape[1]))
				nhx = F.reshape(nhx,(nhx.shape[0],1,nhx.shape[1]))
		
				# TODO(satos) What is the beam search for tree!?
				# now, beam search is ommited.
				nsize = 0
				tree = expand_tree(self.type_size-1,self.embed_root_idx,(nhx,ncx))
				result.append(tree)
			
		return result
